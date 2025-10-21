#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import trimesh
from torch import nn
from loguru import logger
import torch.nn.functional as F
from hugs.models.hugs_wo_trimlp import smpl_lbsmap_top_k, smpl_lbsweight_top_k

from hugs.utils.general import (
    inverse_sigmoid, 
    get_expon_lr_func, 
    strip_symmetric,
    build_scaling_rotation,
)
from hugs.utils.rotations import (
    axis_angle_to_rotation_6d, 
    matrix_to_quaternion, 
    matrix_to_rotation_6d, 
    quaternion_multiply,
    quaternion_to_matrix, 
    rotation_6d_to_axis_angle, 
    rotation_6d_to_matrix,
    torch_rotation_matrix_from_vectors,
)
from hugs.cfg.constants import SMPL_PATH
from hugs.utils.subdivide_smpl import subdivide_smpl_model

from .modules.lbs import lbs_extra
from .modules.smpl_layer import SMPL
from .modules.triplane import TriPlane
from .modules.decoders import AppearanceDecoder, DeformationDecoder, GeometryDecoder


SCALE_Z = 1e-5


class HUGS_TRIMLP:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
        self, 
        sh_degree: int, 
        only_rgb: bool=False,
        n_subdivision: int=0,  
        use_surface=False,  
        init_2d=False,
        rotate_sh=False,
        isotropic=False,
        init_scale_multiplier=0.5,
        n_features=32,
        use_deformer=False,
        disable_posedirs=False,
        triplane_res=256,
        betas=None,
    ):
        self.only_rgb = only_rgb
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self.scaling_multiplier = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.device = 'cuda'
        self.use_surface = use_surface
        self.init_2d = init_2d
        self.rotate_sh = rotate_sh
        self.isotropic = isotropic
        self.init_scale_multiplier = init_scale_multiplier
        self.use_deformer = use_deformer
        self.disable_posedirs = disable_posedirs
        self.cloth_gaussians = None   # will hold cloth xyz + attributes

        self.deformer = 'smpl'
        
        if betas is not None:
            self.create_betas(betas, requires_grad=False)
        
        self.triplane = TriPlane(n_features, resX=triplane_res, resY=triplane_res, resZ=triplane_res).to('cuda')
        self.appearance_dec = AppearanceDecoder(n_features=n_features*3).to('cuda')
        self.deformation_dec = DeformationDecoder(n_features=n_features*3, 
                                                  disable_posedirs=disable_posedirs).to('cuda')
        self.geometry_dec = GeometryDecoder(n_features=n_features*3, use_surface=use_surface).to('cuda')
        
        if n_subdivision > 0:
            logger.info(f"Subdividing SMPL model {n_subdivision} times")
            self.smpl_template = subdivide_smpl_model(smoothing=True, n_iter=n_subdivision).to(self.device)
        else:
            self.smpl_template = SMPL(SMPL_PATH).to(self.device)
            
        self.smpl = SMPL(SMPL_PATH).to(self.device)
            
        edges = trimesh.Trimesh(
            vertices=self.smpl_template.v_template.detach().cpu().numpy(), 
            faces=self.smpl_template.faces, process=False
        ).edges_unique
        self.edges = torch.from_numpy(edges).to(self.device).long()
        
        self.init_values = {}
        self.get_vitruvian_verts()
        
        self.setup_functions()
    
    def initialize_cloth(self, cloth_vertices, cloth_faces=None):
        """Initialize Gaussian parameters for a cloth mesh (e.g., pants, shirt)."""
        device = self.device
        verts = cloth_vertices.to(device)

        # ---- Keep cloth in WORLD SPACE like body (no normalization) ----
        # This ensures triplane queries are consistent between body and cloth
        verts_world = verts  # Keep in world space
        
        logger.info(f"[Cloth Init] vertices range: min={verts_world.min(dim=0)[0]}, max={verts_world.max(dim=0)[0]}")

        # Scaling multiplier (same as SMPL init)
        self.cloth_scaling_multiplier = torch.ones((verts_world.shape[0], 1), device=device)

        # Base SH colors (neutral gray)
        colors = torch.ones_like(verts_world) * 0.5
        shs = torch.zeros((colors.shape[0], 3, 16), dtype=torch.float32, device=device)
        shs[:, :3, 0] = colors
        shs = shs.transpose(1, 2).contiguous()

        # Build mesh if faces are provided
        if cloth_faces is not None:
            mesh = trimesh.Trimesh(
                vertices=verts_world.detach().cpu().numpy(),
                faces=cloth_faces.detach().cpu().numpy(),
                process=False
            )
            vert_normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
            edges = torch.tensor(mesh.edges_unique, dtype=torch.long, device=device)
        else:
            vert_normals = torch.zeros_like(verts_world)
            vert_normals[:, 2] = 1.0
            edges = torch.zeros((0, 2), dtype=torch.long, device=device)

        # Compute per-vertex scales
        scales = torch.zeros_like(verts_world)
        for v in range(verts_world.shape[0]):
            selected_edges = torch.any(edges == v, dim=-1)
            if selected_edges.sum() > 0:
                e = edges[selected_edges]
                edge_len = torch.norm(verts_world[e[:, 0]] - verts_world[e[:, 1]], dim=-1)
                edge_len *= self.init_scale_multiplier
                s = torch.log(torch.max(edge_len))
                scales[v, 0] = s
                scales[v, 1] = s
                if not self.use_surface:
                    scales[v, 2] = s
        scales = torch.exp(scales)

        if self.use_surface or self.init_2d:
            scale_z = torch.ones_like(scales[:, -1:]) * SCALE_Z
            scales = torch.cat([scales[..., :2], scale_z], dim=-1)

        # Align normals → rotation matrices
        gs_normals = torch.zeros_like(vert_normals)
        gs_normals[:, 2] = 1.0
        norm_rotmat = torch_rotation_matrix_from_vectors(gs_normals, vert_normals)
        rot6d = matrix_to_rotation_6d(norm_rotmat)

        # Opacity
        opacity = 0.1 * torch.ones((verts_world.shape[0], 1), dtype=torch.float32, device=device)
       
        # === cloth stats for densification ===
        self.cloth_xyz_gradient_accum = torch.zeros((verts.shape[0], 1), device=device)
        self.cloth_denom = torch.zeros((verts.shape[0], 1), device=device)
        self.cloth_max_radii2D = torch.zeros((verts.shape[0]), device=device)

        # temporary buffers like body
        self.cloth_opacity_tmp = opacity.clone()
        self.cloth_scales_tmp = scales.clone()
        self.cloth_rotmat_tmp = rotation_6d_to_matrix(rot6d).clone()

        # Save into dict
        self.cloth_gaussians = {
            "xyz": nn.Parameter(verts_world.requires_grad_(True)),
            "scales": scales,
            "rot6d_canon": rot6d,
            "shs": shs,
            "opacity": opacity,
            "faces": cloth_faces if cloth_faces is not None else None,
            "normals": vert_normals,
            "edges": edges,
        }

        logger.info(
            f"Initialized cloth gaussians: {verts_world.shape[0]} verts, "
            f"{edges.shape[0]} edges, faces={cloth_faces is not None}"
        )
        
        # Precompute cloth LBS weight table (Option B from user's analysis)
        self.build_cloth_weight_table(verts_world)
    
    @torch.no_grad()
    def build_cloth_weight_table(self, cloth_tpose_verts):
        """
        Precompute cloth LBS weight table by mapping each cloth vertex to nearest SMPL vertex weights.
        This ensures KNN indices during training match the weight table dimensions → no OOB errors.
        
        Args:
            cloth_tpose_verts: Cloth vertices in T-pose/canonical space [N_cloth, 3]
        """
        from hugs.models.hugs_wo_trimlp import smpl_lbsweight_top_k
        
        # Map each cloth T-pose vertex to SMPL template weights
        _, cloth_weights = smpl_lbsweight_top_k(
            lbs_weights=self.smpl.lbs_weights,              # [6890, 24] - SMPL weights
            points=cloth_tpose_verts.unsqueeze(0),          # [1, N_cloth, 3]
            template_points=self.vitruvian_verts.unsqueeze(0),  # [1, 6890, 3] - SMPL template
            K=6,
        )
        
        # Cache as plain attributes (class is not an nn.Module, so no register_buffer)
        self.cloth_lbs_weights_table = cloth_weights.squeeze(0).detach()  # [N_cloth, 24]
        self.cloth_tpose_template = cloth_tpose_verts.clone().detach()    # [N_cloth, 3]
        
        logger.info(
            f"[Cloth LBS] Precomputed weight table: {self.cloth_lbs_weights_table.shape}, "
            f"template: {self.cloth_tpose_template.shape}"
        )
      
    def create_body_pose(self, body_pose, requires_grad=False):
        body_pose = axis_angle_to_rotation_6d(body_pose.reshape(-1, 3)).reshape(-1, 23*6)
        self.body_pose = nn.Parameter(body_pose, requires_grad=requires_grad)
        logger.info(f"Created body pose with shape: {body_pose.shape}, requires_grad: {requires_grad}")
        
    def create_global_orient(self, global_orient, requires_grad=False):
        global_orient = axis_angle_to_rotation_6d(global_orient.reshape(-1, 3)).reshape(-1, 6)
        self.global_orient = nn.Parameter(global_orient, requires_grad=requires_grad)
        logger.info(f"Created global_orient with shape: {global_orient.shape}, requires_grad: {requires_grad}")
        
    def create_betas(self, betas, requires_grad=False):
        self.betas = nn.Parameter(betas, requires_grad=requires_grad)
        logger.info(f"Created betas with shape: {betas.shape}, requires_grad: {requires_grad}")
        
    def create_transl(self, transl, requires_grad=False):
        self.transl = nn.Parameter(transl, requires_grad=requires_grad)
        logger.info(f"Created transl with shape: {transl.shape}, requires_grad: {requires_grad}")
        
    def create_eps_offsets(self, eps_offsets, requires_grad=False):
        logger.info(f"NOT CREATED eps_offsets with shape: {eps_offsets.shape}, requires_grad: {requires_grad}")
    
    @property
    def get_xyz(self):
        return self._xyz
    
    def state_dict(self):
        save_dict = {
            'active_sh_degree': self.active_sh_degree,
            'xyz': self._xyz,
            'triplane': self.triplane.state_dict(),
            'appearance_dec': self.appearance_dec.state_dict(),
            'geometry_dec': self.geometry_dec.state_dict(),
            'deformation_dec': self.deformation_dec.state_dict(),
            'scaling_multiplier': self.scaling_multiplier,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer': self.optimizer.state_dict(),
            'spatial_lr_scale': self.spatial_lr_scale,
        }
        return save_dict
    
    def load_state_dict(self, state_dict, cfg=None):
        self.active_sh_degree = state_dict['active_sh_degree']
        self._xyz = state_dict['xyz']
        self.max_radii2D = state_dict['max_radii2D']
        xyz_gradient_accum = state_dict['xyz_gradient_accum']
        denom = state_dict['denom']
        opt_dict = state_dict['optimizer']
        self.spatial_lr_scale = state_dict['spatial_lr_scale']
        
        self.triplane.load_state_dict(state_dict['triplane'])
        self.appearance_dec.load_state_dict(state_dict['appearance_dec'])
        self.geometry_dec.load_state_dict(state_dict['geometry_dec'])
        self.deformation_dec.load_state_dict(state_dict['deformation_dec'])
        self.scaling_multiplier = state_dict['scaling_multiplier']
        
        if cfg is None:
            from hugs.cfg.config import cfg as default_cfg
            cfg = default_cfg.human.lr
            
        self.setup_optimizer(cfg)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError as e:
            logger.warning(f"Optimizer load failed: {e}")
            logger.warning("Continue without a pretrained optimizer")
            
    def __repr__(self):
        repr_str = "HUGS TRIMLP: \n"
        repr_str += "xyz: {} \n".format(self._xyz.shape)
        repr_str += "max_radii2D: {} \n".format(self.max_radii2D.shape)
        repr_str += "xyz_gradient_accum: {} \n".format(self.xyz_gradient_accum.shape)
        repr_str += "denom: {} \n".format(self.denom.shape)
        return repr_str

    def canon_forward(self):
        tri_feats = self.triplane(self.get_xyz)
        appearance_out = self.appearance_dec(tri_feats)
        geometry_out = self.geometry_dec(tri_feats)
        
        xyz_offsets = geometry_out['xyz']
        gs_rot6d = geometry_out['rotations']
        gs_scales = geometry_out['scales'] * self.scaling_multiplier
        
        gs_opacity = appearance_out['opacity']
        gs_shs = appearance_out['shs'].reshape(-1, 16, 3)
        
        if self.use_deformer:
            deformation_out = self.deformation_dec(tri_feats)
            lbs_weights = deformation_out['lbs_weights']
            lbs_weights = F.softmax(lbs_weights/0.1, dim=-1)
            posedirs = deformation_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None
            
        return {
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'rot6d_canon': gs_rot6d,
            'shs': gs_shs,
            'opacity': gs_opacity,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
        }

    def forward_test(
        self,
        canon_forward_out,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    ):
        xyz_offsets = canon_forward_out['xyz_offsets']
        gs_rot6d = canon_forward_out['rot6d_canon']
        gs_scales = canon_forward_out['scales']
        
        gs_xyz = self.get_xyz + xyz_offsets
        
        gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
        gs_rotq = matrix_to_quaternion(gs_rotmat)

        gs_opacity = canon_forward_out['opacity']
        gs_shs = canon_forward_out['shs'].reshape(-1, 16, 3)
        
        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)
            
        gs_scales_canon = gs_scales.clone()
        
        if self.use_deformer:
            lbs_weights = canon_forward_out['lbs_weights']
            posedirs = canon_forward_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None
        
        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)
        
        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)).reshape(23*3)
            
        if hasattr(self, 'betas') and betas is None:
            betas = self.betas
            
        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]
        
        # vitruvian -> t-pose -> posed
        # remove and reapply the blendshape
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
        )
        
        gt_lbs_weights = None
        if self.use_deformer:
            A_t2pose = smpl_output.A[0]
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian
            deformed_xyz, _, lbs_T, _, _ = lbs_extra(
                A_vitruvian2pose[None], gs_xyz[None], posedirs, lbs_weights, 
                smpl_output.full_pose, disable_posedirs=self.disable_posedirs, pose2rot=True
            )
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)

            with torch.no_grad():
                # gt lbs is needed for lbs regularization loss
                # predicted lbs should be close to gt lbs
                _, gt_lbs_weights = smpl_lbsweight_top_k(
                    lbs_weights=self.smpl.lbs_weights,
                    points=gs_xyz.unsqueeze(0),
                    template_points=self.vitruvian_verts.unsqueeze(0),
                )
                gt_lbs_weights = gt_lbs_weights.squeeze(0)
                if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}")
        else:
            curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
            T_t2pose = smpl_output.T[0]
            T_vitruvian2t = self.inv_T_t2vitruvian.clone()
            T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
            T_vitruvian2pose = T_t2pose @ T_vitruvian2t

            _, lbs_T = smpl_lbsmap_top_k(
                lbs_weights=self.smpl.lbs_weights,
                verts_transform=T_vitruvian2pose.unsqueeze(0),
                points=gs_xyz.unsqueeze(0),
                template_points=self.vitruvian_verts.unsqueeze(0),
                K=6,
            )
            lbs_T = lbs_T.squeeze(0)
        
            homogen_coord = torch.ones_like(gs_xyz[..., :1])
            gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
            deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]
        
        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)
        
        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)
        
        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
        
        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales
            
            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)
        
        self.normals = torch.zeros_like(gs_xyz)
        self.normals[:, 2] = 1.0
        
        canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        
        deformed_gs_shs = gs_shs.clone()
        
        return {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'normals': deformed_normals,
            'normals_canon': canon_normals,
            'active_sh_degree': self.active_sh_degree,
            'rot6d_canon': gs_rot6d,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'gt_lbs_weights': gt_lbs_weights,
        }
         
    def forward_body(
        self,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    ):
        
        tri_feats = self.triplane(self.get_xyz)
        appearance_out = self.appearance_dec(tri_feats)
        geometry_out = self.geometry_dec(tri_feats)
        
        xyz_offsets = geometry_out['xyz']
        gs_rot6d = geometry_out['rotations']
        gs_scales = geometry_out['scales'] * self.scaling_multiplier
        
        gs_xyz = self.get_xyz + xyz_offsets
        
        gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
        gs_rotq = matrix_to_quaternion(gs_rotmat)

        gs_opacity = appearance_out['opacity']
        gs_shs = appearance_out['shs'].reshape(-1, 16, 3)
        
        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)
            
        gs_scales_canon = gs_scales.clone()
        
        if self.use_deformer:
            deformation_out = self.deformation_dec(tri_feats)
            lbs_weights = deformation_out['lbs_weights']
            lbs_weights = F.softmax(lbs_weights/0.1, dim=-1)
            posedirs = deformation_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None
        
        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)
        
        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)).reshape(23*3)
            
        if hasattr(self, 'betas') and betas is None:
            betas = self.betas
            
        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]
        
        # vitruvian -> t-pose -> posed
        # remove and reapply the blendshape
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
        )
        
        gt_lbs_weights = None
        if self.use_deformer:
            A_t2pose = smpl_output.A[0]
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian
            deformed_xyz, _, lbs_T, _, _ = lbs_extra(
                A_vitruvian2pose[None], gs_xyz[None], posedirs, lbs_weights, 
                smpl_output.full_pose, disable_posedirs=self.disable_posedirs, pose2rot=True
            )
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)

            with torch.no_grad():
                # gt lbs is needed for lbs regularization loss
                # predicted lbs should be close to gt lbs
                _, gt_lbs_weights = smpl_lbsweight_top_k(
                    lbs_weights=self.smpl.lbs_weights,
                    points=gs_xyz.unsqueeze(0),
                    template_points=self.vitruvian_verts.unsqueeze(0),
                )
                gt_lbs_weights = gt_lbs_weights.squeeze(0)
                if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}")
        else:
            curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
            T_t2pose = smpl_output.T[0]
            T_vitruvian2t = self.inv_T_t2vitruvian.clone()
            T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
            T_vitruvian2pose = T_t2pose @ T_vitruvian2t

            _, lbs_T = smpl_lbsmap_top_k(
                lbs_weights=self.smpl.lbs_weights,
                verts_transform=T_vitruvian2pose.unsqueeze(0),
                points=gs_xyz.unsqueeze(0),
                template_points=self.vitruvian_verts.unsqueeze(0),
                K=6,
            )
            lbs_T = lbs_T.squeeze(0)
        
            homogen_coord = torch.ones_like(gs_xyz[..., :1])
            gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
            deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]
        
        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)
        
        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)
        
        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
        
        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales
            
            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)
        
        self.normals = torch.zeros_like(gs_xyz)
        self.normals[:, 2] = 1.0
        
        canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        
        deformed_gs_shs = gs_shs.clone()
        
        return {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'normals': deformed_normals,
            'normals_canon': canon_normals,
            'active_sh_degree': self.active_sh_degree,
            'rot6d_canon': gs_rot6d,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'gt_lbs_weights': gt_lbs_weights,
        }

    def forward(
        self,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    ):
        # === body ===
        body_out = self.forward_body(
            global_orient, body_pose, betas, transl, smpl_scale,
            dataset_idx, is_train, ext_tfs
        )

        # === cloth (optional) ===
        cloth_out = None
        if self.cloth_gaussians is not None:
            # Use single-stage approach like human body
            cloth_out = self.forward_cloth(
                global_orient, body_pose, betas, transl, smpl_scale,
                dataset_idx, is_train, ext_tfs
            )

        return {"body": body_out, "cloth": cloth_out} if cloth_out else body_out

    def forward_cloth(
        self,
        global_orient=None,
        body_pose=None,
        betas=None,
        transl=None,
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    ):
        """Single-stage cloth processing (same pattern as human body)."""
        assert self.cloth_gaussians is not None, "Call initialize_cloth first."
        
        # === STEP 1: Feature Extraction (like human body) ===
        cloth_xyz = self.cloth_gaussians["xyz"]  # Cloth Gaussians in canonical space
        tri_feats = self.triplane(cloth_xyz)
        appearance_out = self.appearance_dec(tri_feats)
        geometry_out = self.geometry_dec(tri_feats)
        
        # Extract individual components
        xyz_offsets = geometry_out['xyz']                           # Δμ - position adjustments
        gs_rot6d = geometry_out['rotations']                        # R - rotation adjustments  
        gs_scales = geometry_out['scales'] * self.cloth_scaling_multiplier  # S - scale adjustments
        
        gs_opacity = appearance_out['opacity']                     # o - opacity
        gs_shs = appearance_out['shs'].reshape(-1, 16, 3)         # SH - spherical harmonics colors
        
        # Get canonical cloth position with offsets
        gs_xyz = cloth_xyz + xyz_offsets  # Canonical cloth position + small adjustments
        
        gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
        gs_rotq = matrix_to_quaternion(gs_rotmat)
        
        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)
            
        gs_scales_canon = gs_scales.clone()
        
        # === STEP 2: LBS Weights (like human body) ===
        if self.use_deformer:
            deformation_out = self.deformation_dec(tri_feats)
            lbs_weights = deformation_out['lbs_weights']
            lbs_weights = F.softmax(lbs_weights/0.1, dim=-1)
            posedirs = deformation_out['posedirs']
        else:
            lbs_weights = None
            posedirs = None
        
        # === STEP 3: SMPL Pose Processing (same as human body) ===
        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(self.global_orient[dataset_idx].reshape(-1,6)).reshape(3)
        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(self.body_pose[dataset_idx].reshape(-1,6)).reshape(23*3)
        if hasattr(self, 'betas') and betas is None:
            betas = self.betas
        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]
        # Run SMPL forward pass
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
        )
        # === STEP 4: Deformation (LBS) - SAME AS HUMAN BODY ===
        if self.use_deformer:
            A_t2pose = smpl_output.A[0]
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian
            deformed_xyz, _, lbs_T, _, _ = lbs_extra(
                A_vitruvian2pose[None], gs_xyz[None], posedirs, lbs_weights, 
                smpl_output.full_pose, disable_posedirs=self.disable_posedirs, pose2rot=True
            )
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)
            gt_lbs_weights = None
            # Compute GT LBS weights using precomputed cloth weight table (Option B)
            # KNN now operates on cloth template → indices match cloth_lbs_weights_table → no OOB!
            with torch.no_grad():
                try:
                    from hugs.models.hugs_wo_trimlp import smpl_lbsweight_top_k
                    
                    # Use precomputed cloth weight table with original cloth template
                    # We want to map current cloth vertices (gs_xyz) to original cloth template positions
                    # This gives us consistent LBS weights based on the original cloth topology
                    _, gt_lbs_weights = smpl_lbsweight_top_k(
                        lbs_weights=self.cloth_lbs_weights_table,     # [N_cloth, 24] - precomputed!
                        points=gs_xyz.unsqueeze(0),                   # [1, N_cloth, 3] - current cloth vertices
                        template_points=self.cloth_tpose_template.unsqueeze(0),  # [1, N_cloth, 3] - original template
                        K=6,
                    )
                    gt_lbs_weights = gt_lbs_weights.squeeze(0)  # [N_cloth, 24]
                    
                    # Validate
                    weight_sum = gt_lbs_weights.sum(-1).mean().item()
                    if abs(weight_sum - 1.0) > 1e-5:
                        from loguru import logger
                        logger.warning(f"Cloth GT LBS weights sum: {weight_sum:.6f} (expected 1.0)")
                        
                except Exception as e:
                    from loguru import logger
                    logger.error(f"Cloth LBS weight computation failed: {e}")
                    gt_lbs_weights = None
        else:
            curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
            T_t2pose = smpl_output.T[0]
            T_vitruvian2t = self.inv_T_t2vitruvian.clone()
            T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
            T_vitruvian2pose = T_t2pose @ T_vitruvian2t
            
            _, lbs_T = smpl_lbsmap_top_k(
                lbs_weights=self.smpl.lbs_weights,
                verts_transform=T_vitruvian2pose.unsqueeze(0),
                points=gs_xyz.unsqueeze(0),
                template_points=self.vitruvian_verts.unsqueeze(0),
                K=6,
            )
            lbs_T = lbs_T.squeeze(0)
            
            homogen_coord = torch.ones_like(gs_xyz[..., :1])
            gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
            deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]
        
        # === STEP 5: Final Transforms (same as human body) ===
        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)
        
        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)
        
        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
        
        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales
            
            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)
        
        # === STEP 6: Compute Normals (same as human body) ===
        normals = torch.zeros_like(gs_xyz)
        normals[:, 2] = 1.0
        canon_normals = (gs_rotmat @ normals.unsqueeze(-1)).squeeze(-1)
        deformed_normals = (deformed_gs_rotmat @ normals.unsqueeze(-1)).squeeze(-1)
        
        # Debug logging
        if is_train and torch.rand(1) < 0.01:  # Log 1% of the time
            from loguru import logger
            deformation_mag = torch.norm(deformed_xyz - gs_xyz, dim=1).mean()
            logger.info(f"Cloth single-stage deformation magnitude: {deformation_mag:.6f}")
        
        return {
            'xyz': deformed_xyz,           # Final deformed cloth vertices
            'xyz_canon': gs_xyz,           # Canonical cloth vertices
            'xyz_offsets': xyz_offsets,    # Position offsets
            'scales': gs_scales,           # Final scales
            'scales_canon': gs_scales_canon, # Canonical scales
            'rotq': deformed_gs_rotq,      # Final rotations
            'rotq_canon': gs_rotq,         # Canonical rotations
            'rotmat': deformed_gs_rotmat,  # Final rotation matrices
            'rotmat_canon': gs_rotmat,     # Canonical rotation matrices
            'shs': gs_shs,                # Colors
            'opacity': gs_opacity,         # Opacity
            'normals': deformed_normals,   # Final normals
            'normals_canon': canon_normals, # Canonical normals
            'lbs_weights': lbs_weights,    # LBS weights used
            'gt_lbs_weights': gt_lbs_weights, # GT LBS weights for regularization
            'posedirs': posedirs,          # Pose directions used
            'active_sh_degree': self.active_sh_degree,
            'rot6d_canon': gs_rot6d,
        }

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            logger.info(f"Going from SH degree {self.active_sh_degree} to {self.active_sh_degree + 1}")
            self.active_sh_degree += 1

    @torch.no_grad()
    def get_vitruvian_verts(self):
        vitruvian_pose = torch.zeros(69, dtype=self.smpl.dtype, device=self.device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = self.smpl(body_pose=vitruvian_pose[None], betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = smpl_output.vertices[0]
        self.A_t2vitruvian = smpl_output.A[0].detach()
        self.T_t2vitruvian = smpl_output.T[0].detach()
        self.inv_T_t2vitruvian = torch.inverse(self.T_t2vitruvian)
        self.inv_A_t2vitruvian = torch.inverse(self.A_t2vitruvian)
        self.canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
        self.canonical_offsets = self.canonical_offsets[0].detach()
        self.vitruvian_verts = vitruvian_verts.detach()
        return vitruvian_verts.detach()
    
    @torch.no_grad()
    def get_vitruvian_verts_template(self):
        vitruvian_pose = torch.zeros(69, dtype=self.smpl_template.dtype, device=self.device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = self.smpl_template(body_pose=vitruvian_pose[None], betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = smpl_output.vertices[0]
        return vitruvian_verts.detach()
    
        @torch.no_grad()
        def get_tpose_verts_template(self):
            """Get T-pose vertices from ORIGINAL SMPL model (not subdivided)."""
            # Use original SMPL model for LBS weight computation (not subdivided version)
            original_smpl = SMPL(SMPL_PATH).to(self.device)
            tpose_pose = torch.zeros(69, dtype=original_smpl.dtype, device=self.device)  # All zeros = T-pose
            smpl_output = original_smpl(body_pose=tpose_pose[None], betas=self.betas[None], disable_posedirs=False)
            tpose_verts = smpl_output.vertices[0]  # This will be [6890, 3] - original SMPL template vertices
            return tpose_verts.detach()
    
    def train(self):
        pass
    
    def eval(self):
        pass
    
    def initialize(self):
        t_pose_verts = self.get_vitruvian_verts_template()
        
        self.scaling_multiplier = torch.ones((t_pose_verts.shape[0], 1), device="cuda")
        
        xyz_offsets = torch.zeros_like(t_pose_verts)
        colors = torch.ones_like(t_pose_verts) * 0.5
        
        shs = torch.zeros((colors.shape[0], 3, 16)).float().cuda()
        shs[:, :3, 0 ] = colors
        shs[:, 3:, 1:] = 0.0
        shs = shs.transpose(1, 2).contiguous()
        
        scales = torch.zeros_like(t_pose_verts)
        for v in range(t_pose_verts.shape[0]):
            selected_edges = torch.any(self.edges == v, dim=-1)
            selected_edges_len = torch.norm(
                t_pose_verts[self.edges[selected_edges][0]] - t_pose_verts[self.edges[selected_edges][1]], 
                dim=-1
            )
            selected_edges_len *= self.init_scale_multiplier
            scales[v, 0] = torch.log(torch.max(selected_edges_len))
            scales[v, 1] = torch.log(torch.max(selected_edges_len))
            
            if not self.use_surface:
                scales[v, 2] = torch.log(torch.max(selected_edges_len))
        
        if self.use_surface or self.init_2d:
            scales = scales[..., :2]
            
        scales = torch.exp(scales)
        
        if self.use_surface or self.init_2d:
            scale_z = torch.ones_like(scales[:, -1:]) * SCALE_Z
            scales = torch.cat([scales, scale_z], dim=-1)
        
        import trimesh
        mesh = trimesh.Trimesh(vertices=t_pose_verts.detach().cpu().numpy(), faces=self.smpl_template.faces)
        vert_normals = torch.tensor(mesh.vertex_normals).float().cuda()
        
        gs_normals = torch.zeros_like(vert_normals)
        gs_normals[:, 2] = 1.0
        
        norm_rotmat = torch_rotation_matrix_from_vectors(gs_normals, vert_normals)

        rotq = matrix_to_quaternion(norm_rotmat)
        rot6d = matrix_to_rotation_6d(norm_rotmat)
                
        self.normals = gs_normals
        deformed_normals = (norm_rotmat @ gs_normals.unsqueeze(-1)).squeeze(-1)
        
        opacity = 0.1 * torch.ones((t_pose_verts.shape[0], 1), dtype=torch.float, device="cuda")
        
        posedirs = self.smpl_template.posedirs.detach().clone()
        lbs_weights = self.smpl_template.lbs_weights.detach().clone()

        self.n_gs = t_pose_verts.shape[0]
        self._xyz = nn.Parameter(t_pose_verts.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        return {
            'xyz_offsets': xyz_offsets,
            'scales': scales,
            'rot6d_canon': rot6d,
            'shs': shs,
            'opacity': opacity,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'deformed_normals': deformed_normals,
            'faces': self.smpl.faces_tensor,
            'edges': self.edges,
        }

    def setup_optimizer(self, cfg):
        self.percent_dense = cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.spatial_lr_scale = cfg.smpl_spatial
        
        params = [
            {'params': [self._xyz], 'lr': cfg.position_init * cfg.smpl_spatial, "name": "xyz"},
            {'params': self.triplane.parameters(), 'lr': cfg.vembed, 'name': 'v_embed'},
            {'params': self.geometry_dec.parameters(), 'lr': cfg.geometry, 'name': 'geometry_dec'},
            {'params': self.appearance_dec.parameters(), 'lr': cfg.appearance, 'name': 'appearance_dec'},
            {'params': self.deformation_dec.parameters(), 'lr': cfg.deformation, 'name': 'deform_dec'},
        ]
        
        if hasattr(self, 'global_orient') and self.global_orient.requires_grad:
            params.append({'params': self.global_orient, 'lr': cfg.smpl_pose, 'name': 'global_orient'})
        
        if hasattr(self, 'body_pose') and self.body_pose.requires_grad:
            params.append({'params': self.body_pose, 'lr': cfg.smpl_pose, 'name': 'body_pose'})
            
        if hasattr(self, 'betas') and self.betas.requires_grad:
            params.append({'params': self.betas, 'lr': cfg.smpl_betas, 'name': 'betas'})
            
        if hasattr(self, 'transl') and self.transl.requires_grad:
            params.append({'params': self.transl, 'lr': cfg.smpl_trans, 'name': 'transl'})
        if self.cloth_gaussians is not None:
            params.append({'params': [self.cloth_gaussians["xyz"]], 'lr': cfg.position_init * cfg.smpl_spatial, 'name': 'cloth_xyz'})   # match expected key name

        self.non_densify_params_keys = [
            'global_orient', 'body_pose', 'betas', 'transl', 
            'v_embed', 'geometry_dec', 'appearance_dec', 'deform_dec',
        ]
        
        for param in params:
            logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=cfg.position_init  * cfg.smpl_spatial,
            lr_final=cfg.position_final  * cfg.smpl_spatial,
            lr_delay_mult=cfg.position_delay_mult,
            max_steps=cfg.position_max_steps,
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        lr = None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["xyz", "cloth_xyz"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
        return lr



    def replace_tensor_to_optimizer(self, tensor, name):
        for group in self.optimizer.param_groups:
            if group.get("name") == name:
                old_param = group["params"][0]
                # Preserve optimizer state if exists
                if old_param in self.optimizer.state:
                    stored_state = self.optimizer.state[old_param]
                    self.optimizer.state.pop(old_param)
                    self.optimizer.state[tensor] = {
                        k: v.clone() for k, v in stored_state.items()
                    }
                else:
                    # 🔧 initialize fresh Adam state if not present
                    self.optimizer.state[tensor] = {
                        "step": torch.zeros(1, device=tensor.device),
                        "exp_avg": torch.zeros_like(tensor),
                        "exp_avg_sq": torch.zeros_like(tensor),
                    }
                group["params"][0] = tensor
                return {name: tensor}
        raise ValueError(f"Param group {name} not found in optimizer")


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue
            if group["name"] == "cloth_xyz":
                # IMPORTANT: skip cloth when pruning body
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self.scaling_multiplier = self.scaling_multiplier[valid_points_mask]
        
        self.scales_tmp = self.scales_tmp[valid_points_mask]
        self.opacity_tmp = self.opacity_tmp[valid_points_mask]
        self.rotmat_tmp = self.rotmat_tmp[valid_points_mask]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue

            # >>> ADD THIS GUARD <<<
            if group["name"] not in tensors_dict:
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"]   = torch.cat(
                    (stored_state["exp_avg"],   torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                old_param = group["params"][0]
                del self.optimizer.state[old_param]
                group["params"][0] = nn.Parameter(
                    torch.cat((old_param, extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(self, new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp):
        d = {
            "xyz": new_xyz,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self.scaling_multiplier = torch.cat((self.scaling_multiplier, new_scaling_multiplier), dim=0)
        self.opacity_tmp = torch.cat([self.opacity_tmp, new_opacity_tmp], dim=0)
        self.scales_tmp = torch.cat([self.scales_tmp, new_scales_tmp], dim=0)
        self.rotmat_tmp = torch.cat([self.rotmat_tmp, new_rotmat_tmp], dim=0)
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        scales = self.scales_tmp
        rotation = self.rotmat_tmp
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scales, dim=1).values > self.percent_dense*scene_extent)
        # filter elongated gaussians
        med = scales.median(dim=1, keepdim=True).values 
        stdmed_mask = (((scales - med) / med).squeeze(-1) >= 1.0).any(dim=-1)
        selected_pts_mask = torch.logical_and(selected_pts_mask, stdmed_mask)
        
        stds = scales[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=torch.relu(stds))
        rots = rotation[selected_pts_mask].repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask].repeat(N,1) / (0.8*N)
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask].repeat(N,1)
        new_scales_tmp = self.scales_tmp[selected_pts_mask].repeat(N,1)
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask].repeat(N,1,1)
        
        self.densification_postfix(new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        scales = self.scales_tmp
        grad_cond = torch.norm(grads, dim=-1) >= grad_threshold
        scale_cond = torch.max(scales, dim=1).values <= self.percent_dense*scene_extent
        
        selected_pts_mask = torch.where(grad_cond, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_cond)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask]
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask]
        new_scales_tmp = self.scales_tmp[selected_pts_mask]
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp)

    def densify_and_prune(self, human_gs_out, max_grad, min_opacity, extent, max_screen_size, max_n_gs=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.opacity_tmp = human_gs_out['opacity']
        self.scales_tmp = human_gs_out['scales_canon']
        self.rotmat_tmp = human_gs_out['rotmat_canon']
        
        max_n_gs = max_n_gs if max_n_gs else self.get_xyz.shape[0] + 1
        
        if self.get_xyz.shape[0] <= max_n_gs:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.opacity_tmp < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scales_tmp.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.n_gs = self.get_xyz.shape[0]
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[:update_filter.shape[0]][update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    def add_cloth_densification_stats(self, viewspace_tensor, update_filter):
        self.cloth_xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_tensor.grad[:update_filter.shape[0]][update_filter, :2], dim=-1, keepdim=True
        )
        self.cloth_denom[update_filter] += 1
    def _ensure_cloth_group_exists(self, lr):
        # only add if not present
        names = [g["name"] for g in self.optimizer.param_groups]
        if "cloth_xyz" not in names:
            self.optimizer.add_param_group({
                "params": [self.cloth_gaussians["xyz"]],
                "lr": lr,
                "name": "cloth_xyz",
            })


    def cloth_densification_postfix(self, new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp):
        d = {"cloth_xyz": new_xyz}
        optimizable = self.cat_tensors_to_optimizer(d)
        self.cloth_gaussians["xyz"] = optimizable["cloth_xyz"]

        self.cloth_scaling_multiplier = torch.cat((self.cloth_scaling_multiplier, new_scaling_multiplier), dim=0)
        self.cloth_opacity_tmp = torch.cat([self.cloth_opacity_tmp, new_opacity_tmp], dim=0)
        self.cloth_scales_tmp = torch.cat([self.cloth_scales_tmp, new_scales_tmp], dim=0)
        self.cloth_rotmat_tmp = torch.cat([self.cloth_rotmat_tmp, new_rotmat_tmp], dim=0)

        self.cloth_xyz_gradient_accum = torch.zeros((self.cloth_gaussians["xyz"].shape[0], 1), device="cuda")
        self.cloth_denom = torch.zeros((self.cloth_gaussians["xyz"].shape[0], 1), device="cuda")
        self.cloth_max_radii2D = torch.zeros((self.cloth_gaussians["xyz"].shape[0]), device="cuda")

    def cloth_densify_and_clone(self, grads, grad_threshold, scene_extent):
        grad_cond = torch.norm(grads, dim=-1) >= grad_threshold
        scale_cond = torch.max(self.cloth_scales_tmp, dim=1).values <= self.percent_dense * scene_extent
        mask = grad_cond & scale_cond

        new_xyz = self.cloth_gaussians["xyz"][mask]
        new_scaling_multiplier = self.cloth_scaling_multiplier[mask]
        new_opacity_tmp = self.cloth_opacity_tmp[mask]
        new_scales_tmp = self.cloth_scales_tmp[mask]
        new_rotmat_tmp = self.cloth_rotmat_tmp[mask]

        self.cloth_densification_postfix(new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp)

    def cloth_densify_and_prune(self, cloth_gs_out, grad_threshold, min_opacity, extent, max_screen_size, max_n_gs=None):
        grads = self.cloth_xyz_gradient_accum / self.cloth_denom
        grads[grads.isnan()] = 0.0

        self.cloth_opacity_tmp = cloth_gs_out["opacity"]
        self.cloth_scales_tmp = cloth_gs_out["scales_canon"]
        self.cloth_rotmat_tmp = cloth_gs_out["rotmat_canon"]

        max_n_gs = max_n_gs if max_n_gs else self.cloth_gaussians["xyz"].shape[0] + 1
        if self.cloth_gaussians["xyz"].shape[0] <= max_n_gs:
            self.cloth_densify_and_clone(grads, grad_threshold, extent)

        # --- create prune mask ---
        prune_mask = (self.cloth_opacity_tmp < min_opacity).squeeze()

        if max_screen_size:
            big_vs = self.cloth_max_radii2D > max_screen_size
            big_ws = self.cloth_scales_tmp.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_vs | big_ws

        # --- ensure same length as cloth xyz ---
        current_n = self.cloth_gaussians["xyz"].shape[0]
        if prune_mask.shape[0] != current_n:
            prune_mask = prune_mask[:current_n]

        # apply pruning
        keep = ~prune_mask
        old_cloth_xyz = self.cloth_gaussians["xyz"]
        new_cloth_xyz = nn.Parameter(old_cloth_xyz[keep].requires_grad_(True))
        self.cloth_gaussians["xyz"] = new_cloth_xyz

        # Replace the cloth param in the optimizer while preserving state
        if self.optimizer is None:
            logger.warning("Optimizer not set up yet; skipping optimizer replacement for cloth_xyz")
        else:
            # Find the cloth_xyz parameter group
            cloth_group = None
            for group in self.optimizer.param_groups:
                if group.get("name") == "cloth_xyz":
                    cloth_group = group
                    break
            
            if cloth_group is not None:
                old_param = cloth_group["params"][0]
                
                # Preserve optimizer state for kept parameters
                if old_param in self.optimizer.state:
                    stored_state = self.optimizer.state[old_param]
                    
                    # Filter state tensors to match kept indices
                    for key, value in stored_state.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) > 0 and value.shape[0] == old_param.shape[0]:
                            stored_state[key] = value[keep]
                    
                    # Remove old state and add new state
                    del self.optimizer.state[old_param]
                    self.optimizer.state[new_cloth_xyz] = stored_state
                else:
                    # Initialize fresh state for new parameter
                    self.optimizer.state[new_cloth_xyz] = {
                        "step": torch.zeros(1, device=new_cloth_xyz.device),
                        "exp_avg": torch.zeros_like(new_cloth_xyz),
                        "exp_avg_sq": torch.zeros_like(new_cloth_xyz),
                    }
                
                # Update the parameter group
                cloth_group["params"][0] = new_cloth_xyz
                logger.info(f"Updated cloth_xyz optimizer state: {old_param.shape} -> {new_cloth_xyz.shape}")
            else:
                logger.warning("cloth_xyz parameter group not found in optimizer")

        # Now prune your side buffers
        self.cloth_scaling_multiplier = self.cloth_scaling_multiplier[keep]
        self.cloth_opacity_tmp = self.cloth_opacity_tmp[keep]
        self.cloth_scales_tmp = self.cloth_scales_tmp[keep]
        self.cloth_rotmat_tmp = self.cloth_rotmat_tmp[keep]

        self.cloth_xyz_gradient_accum = self.cloth_xyz_gradient_accum[keep]
        self.cloth_denom = self.cloth_denom[keep]
        self.cloth_max_radii2D = self.cloth_max_radii2D[keep]
        
        # CRITICAL: Update cloth LBS weight table and template after densification
        # This prevents OOB errors when vertex count changes
        self.cloth_lbs_weights_table = self.cloth_lbs_weights_table[keep]
        self.cloth_tpose_template = self.cloth_tpose_template[keep]
        logger.info(f"Updated cloth LBS table after pruning: {self.cloth_lbs_weights_table.shape}")

        # Remap cloth edges/faces to the new compact vertex indexing so ARAP stays valid
        old_n = keep.shape[0]
        index_map = torch.full((old_n,), -1, device=keep.device, dtype=torch.long)
        index_map[keep] = torch.arange(keep.sum(), device=keep.device, dtype=torch.long)

        # Update edges if present
        if "edges" in self.cloth_gaussians and self.cloth_gaussians["edges"] is not None and self.cloth_gaussians["edges"].numel() > 0:
            edges_old = self.cloth_gaussians["edges"]
            edges_new = index_map[edges_old]
            valid_edges = (edges_new[:, 0] >= 0) & (edges_new[:, 1] >= 0)
            self.cloth_gaussians["edges"] = edges_new[valid_edges]
            logger.info(f"Remapped cloth edges: {edges_old.shape[0]} -> {self.cloth_gaussians['edges'].shape[0]}")

        # Update faces if present
        if "faces" in self.cloth_gaussians and self.cloth_gaussians["faces"] is not None and self.cloth_gaussians["faces"].numel() > 0:
            faces_old = self.cloth_gaussians["faces"]
            faces_new = index_map[faces_old]
            valid_faces = (faces_new >= 0).all(dim=1)
            self.cloth_gaussians["faces"] = faces_new[valid_faces]
            logger.info(f"Remapped cloth faces: {faces_old.shape[0]} -> {self.cloth_gaussians['faces'].shape[0]}")
