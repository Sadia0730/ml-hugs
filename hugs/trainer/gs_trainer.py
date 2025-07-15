#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import glob
import shutil
import torch
import itertools
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from lpips import LPIPS
from loguru import logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from hugs.datasets.utils import (
    get_rotating_camera,
    get_smpl_canon_params,
    get_smpl_static_params, 
    get_static_camera
)
from hugs.losses.utils import ssim
from hugs.datasets import NeumanDataset
from hugs.losses.loss import HumanSceneLoss
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models.hugs_wo_trimlp import HUGS_WO_TRIMLP
from hugs.models import SceneGS
from hugs.utils.init_opt import optimize_init
from hugs.renderer.gs_renderer import render_human_scene
from hugs.utils.vis import save_ply
from hugs.utils.image import psnr, save_image
from hugs.utils.general import RandomIndexIterator, load_human_ckpt, save_images, create_video
from hugs.utils.distributed import convert_model_to_ddp


def get_train_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-train')
        dataset = NeumanDataset(
            cfg.dataset.seq, 'train', 
            render_mode=cfg.mode,
            add_bg_points=cfg.scene.add_bg_points,
            num_bg_points=cfg.scene.num_bg_points,
            bg_sphere_dist=cfg.scene.bg_sphere_dist,
            clean_pcd=cfg.scene.clean_pcd,
        )
    
    return dataset


def get_val_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-val')
        dataset = NeumanDataset(cfg.dataset.seq, 'val', cfg.mode)
   
    return dataset


def get_anim_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-anim')
        dataset = NeumanDataset(cfg.dataset.seq, 'anim', cfg.mode)
    elif cfg.dataset.name == 'zju':
        dataset = None
        
    return dataset


class GaussianTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.human_gs, self.scene_gs = None, None
        
        if cfg.mode in ['human', 'human_scene']:
            # Define valid constructor parameters for both model types
            valid_hugs_params = {
                'sh_degree', 'only_rgb', 'n_subdivision', 'use_surface', 'init_2d', 
                'rotate_sh', 'isotropic', 'init_scale_multiplier', 'n_features', 
                'use_deformer', 'disable_posedirs', 'triplane_res', 'betas'
            }
            
            # Filter configuration to only include valid constructor parameters
            human_config = {k: v for k, v in cfg.human.items() if k in valid_hugs_params}
            
            if cfg.mode == 'human_scene' and not cfg.human.use_trimlp:
                self.human_gs = HUGS_WO_TRIMLP(**human_config).to(self.device)
            else:
                self.human_gs = HUGS_TRIMLP(**human_config).to(self.device)
            logger.info(f"Human Gaussian model created")
            

            
        if cfg.mode in ['scene', 'human_scene']:
            # Define valid constructor parameters for SceneGS
            valid_scene_params = {'sh_degree', 'only_rgb'}
            
            # Filter configuration to only include valid constructor parameters
            scene_config = {k: v for k, v in cfg.scene.items() if k in valid_scene_params}
            
            self.scene_gs = SceneGS(**scene_config).to(self.device)
            if cfg.world_size > 1:
                logger.info(f"Wrapping scene model in DDP on device {self.device}")
                self.scene_gs = convert_model_to_ddp(self.scene_gs, self.device)
            logger.info(f"Scene Gaussian model created")
        
        # CRITICAL: Initialize the human model BEFORE DDP wrapping
        if self.human_gs:
            logger.info("Initializing human Gaussian model...")
            init_values = self.human_gs.initialize()
            self.human_gs.setup_optimizer(cfg.human.lr)
            logger.info(f"Human model initialized with {init_values['xyz_offsets'].shape[0]} Gaussians")
            
            # Now wrap in DDP if distributed - AFTER initialization
            if cfg.world_size > 1:
                logger.info(f"Wrapping initialized human model in DDP on device {self.device}")
                self.human_gs = convert_model_to_ddp(self.human_gs, self.device)
        
        logger.info(self.human_gs)
        
        # Initialize datasets
        self.train_dataset = get_train_dataset(cfg)
        self.val_dataset = get_val_dataset(cfg)
        
        # Initialize loss function
        self.loss_fn = HumanSceneLoss(
            l_ssim_w=cfg.human.loss.ssim_w,
            l_l1_w=cfg.human.loss.l1_w,
            l_lpips_w=cfg.human.loss.lpips_w,
            l_lbs_w=cfg.human.loss.lbs_w,
            l_humansep_w=cfg.human.loss.humansep_w,
            num_patches=cfg.human.loss.num_patches,
            patch_size=cfg.human.loss.patch_size,
            use_patches=cfg.human.loss.use_patches,
            bg_color=cfg.bg_color,
        )

    def train(self):
        # if self.human_gs:
        #     self.human_gs.train()

        # DataLoader with DistributedSampler
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.cfg.world_size,
            rank=self.cfg.rank,
            shuffle=True
        )
        loader = DataLoader(self.train_dataset, sampler=sampler, batch_size=1, num_workers=getattr(self.cfg.train, 'num_workers', 0), drop_last=True)

        num_epochs = (self.cfg.train.num_steps // len(loader)) + 1
        t_iter = 0
        print(f"num_workers: {getattr(self.cfg.train, 'num_workers', 0)}")
        print(f"rank: {self.cfg.rank}")
        print(f"world_size: {self.cfg.world_size}")
        print(f"num_epochs: {num_epochs}")
        print(f"dataset size: {len(self.train_dataset)}")
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            if self.cfg.rank == 0:
                pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            else:
                pbar = loader
            for data in pbar:
                if t_iter > self.cfg.train.num_steps:
                    break
                render_mode = self.cfg.mode
                if self.scene_gs and self.cfg.train.optim_scene:
                    if hasattr(self.scene_gs, 'module'):
                        self.scene_gs.module.update_learning_rate(t_iter)
                    else:
                        self.scene_gs.update_learning_rate(t_iter)
                        
                if self.human_gs:
                    model = self.human_gs.module if hasattr(self.human_gs, 'module') else self.human_gs
                    if hasattr(model, 'update_learning_rate'):
                        model.update_learning_rate(t_iter)
                data = data[0] if isinstance(data, list) or isinstance(data, tuple) else data
                
                # CRITICAL FIX: Extract dataset_idx safely and handle tensor conversion
                if 'dataset_idx' in data:
                    idx_tensor = data.pop('dataset_idx')
                    dataset_idx = int(idx_tensor.item()) if torch.is_tensor(idx_tensor) else int(idx_tensor)
                else:
                    dataset_idx = -1  # Default fallback
                
                # CRITICAL FIX: Squeeze batch dimensions carefully to avoid breaking tensor shapes
                for k, v in list(data.items()):
                    if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == 1:
                        # Only squeeze the first dimension if it's size 1
                        data[k] = v.squeeze(0)
                
                human_gs_out, scene_gs_out = None, None
                if self.human_gs:
                    # CRITICAL FIX: Pass all required parameters to the forward method
                    human_gs_out = self.human_gs.forward(
                        global_orient=data.get('global_orient', None),
                        body_pose=data.get('body_pose', None), 
                        betas=data.get('betas', None),
                        transl=data.get('transl', None),
                        smpl_scale=data['smpl_scale'][None],
                        dataset_idx=dataset_idx,
                        is_train=True,
                        ext_tfs=None,
                    )
                if self.scene_gs:
                    if t_iter >= self.cfg.scene.opt_start_iter:
                        scene_gs_out = self.scene_gs.forward()
                    else:
                        render_mode = 'human'
                bg_color = torch.rand(3, dtype=torch.float32, device=self.cfg.device)
                if self.cfg.human.loss.humansep_w > 0.0 and render_mode == 'human_scene':
                    render_human_separate = True
                    human_bg_color = torch.rand(3, dtype=torch.float32, device=self.cfg.device)
                else:
                    human_bg_color = None
                    render_human_separate = False
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=bg_color,
                    human_bg_color=human_bg_color,
                    render_mode=render_mode,
                    render_human_separate=render_human_separate,
                )
                if self.human_gs:
                    model = self.human_gs.module if hasattr(self.human_gs, 'module') else self.human_gs
                    model.init_values['edges'] = model.edges
                human_model = self.human_gs.module if hasattr(self.human_gs, 'module') else self.human_gs if self.human_gs else None
                loss, loss_dict, loss_extras = self.loss_fn(
                    data,
                    render_pkg,
                    human_gs_out,
                    render_mode=render_mode,
                    human_gs_init_values=human_model.init_values if human_model else None,
                    bg_color=bg_color,
                    human_bg_color=human_bg_color,
                )
                model = self.human_gs.module if hasattr(self.human_gs, 'module') else self.human_gs if self.human_gs else None
                if model and hasattr(model, 'loss_nonrigid_reg'):
                    nonrigid_reg_loss = model.loss_nonrigid_reg
                    nonrigid_smooth_loss = model.loss_nonrigid_smooth
                    nonrigid_delta_mag = torch.norm(human_gs_out['nonrigid_delta'], dim=-1).mean()
                    loss_dict['nonrigid_reg'] = nonrigid_reg_loss
                    loss_dict['nonrigid_smooth'] = nonrigid_smooth_loss
                    loss_dict['nonrigid_delta_mag'] = nonrigid_delta_mag
                    loss = loss + self.cfg.human.loss.nonrigid_w * nonrigid_reg_loss
                    if hasattr(self.cfg.human.loss, 'nonrigid_smooth_w'):
                        loss = loss + self.cfg.human.loss.nonrigid_smooth_w * nonrigid_smooth_loss
                loss.backward()
                loss_dict['loss'] = loss
                # --- Densification logic (all ranks, but guard file writes) ---
                if self.scene_gs and t_iter >= self.cfg.scene.opt_start_iter:
                    if (t_iter - self.cfg.scene.opt_start_iter) < self.cfg.scene.densify_until_iter and self.cfg.mode in ['scene', 'human_scene']:
                        render_pkg['scene_viewspace_points'] = render_pkg['viewspace_points']
                        render_pkg['scene_viewspace_points'].grad = render_pkg['viewspace_points'].grad
                        with torch.no_grad():
                            self.scene_densification(
                                visibility_filter=render_pkg['scene_visibility_filter'],
                                radii=render_pkg['scene_radii'],
                                viewspace_point_tensor=render_pkg['scene_viewspace_points'],
                                iteration=(t_iter - self.cfg.scene.opt_start_iter) + 1,
                            )
                if self.human_gs and t_iter < self.cfg.human.densify_until_iter and self.cfg.mode in ['human', 'human_scene']:
                    render_pkg['human_viewspace_points'] = render_pkg['viewspace_points'][:human_gs_out['xyz'].shape[0]]
                    render_pkg['human_viewspace_points'].grad = render_pkg['viewspace_points'].grad[:human_gs_out['xyz'].shape[0]]
                    with torch.no_grad():
                        self.human_densification(
                            human_gs_out=human_gs_out,
                            visibility_filter=render_pkg['human_visibility_filter'],
                            radii=render_pkg['human_radii'],
                            viewspace_point_tensor=render_pkg['human_viewspace_points'],
                            iteration=t_iter+1,
                        )
                # --- SH degree scheduling (all ranks, but .module for DDP) ---
                if t_iter % 1000 == 0:
                    if self.human_gs:
                        if hasattr(self.human_gs, 'module'):
                            self.human_gs.module.oneupSHdegree()
                        else:
                            self.human_gs.oneupSHdegree()
                    if self.scene_gs:
                        if hasattr(self.scene_gs, 'module'):
                            self.scene_gs.module.oneupSHdegree()
                        else:
                            self.scene_gs.oneupSHdegree()
                # --- Progress bar, checkpointing, validation, animation (rank 0 only) ---
                if self.cfg.rank == 0 and t_iter % 10 == 0:
                    human_model = self.human_gs.module if hasattr(self.human_gs, 'module') else self.human_gs if self.human_gs else None
                    scene_model = self.scene_gs.module if hasattr(self.scene_gs, 'module') else self.scene_gs if self.scene_gs else None
                    
                    postfix_dict = {
                        "#hp": f"{human_model.n_gs/1000 if human_model and hasattr(human_model, 'n_gs') else 0:.1f}K",
                        "#sp": f"{scene_model.get_xyz.shape[0]/1000 if scene_model and hasattr(scene_model, 'get_xyz') else 0:.1f}K",
                        'h_sh_d': human_model.active_sh_degree if human_model and hasattr(human_model, 'active_sh_degree') else 0,
                        's_sh_d': scene_model.active_sh_degree if scene_model and hasattr(scene_model, 'active_sh_degree') else 0,
                    }
                    for k, v in loss_dict.items():
                        postfix_dict["l_"+k] = f"{v.item():.4f}"
                    if hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix(postfix_dict)
                    if t_iter == self.cfg.train.num_steps:
                        if hasattr(pbar, 'close'):
                            pbar.close()
                if self.cfg.rank == 0 and t_iter % self.cfg.train.save_ckpt_interval == 0 and t_iter > 0:
                    self.save_ckpt(t_iter)
                if self.cfg.rank == 0 and t_iter % self.cfg.train.val_interval == 0 and t_iter > 0:
                    self.validate(t_iter)
                if self.cfg.rank == 0 and t_iter % self.cfg.train.anim_interval == 0 and t_iter > 0 and self.cfg.train.anim_interval > 0:
                    if self.human_gs:
                        save_ply(human_gs_out, f'{self.cfg.logdir}/meshes/human_{t_iter:06d}_splat.ply')
                    if self.anim_dataset is not None:
                        self.animate(t_iter)
                    if self.cfg.mode in ['human', 'human_scene']:
                        self.render_canonical(t_iter, nframes=self.cfg.human.canon_nframes)
                if self.human_gs:
                    model = self.human_gs.module if hasattr(self.human_gs, 'module') else self.human_gs
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none=True)
                if self.scene_gs and self.cfg.train.optim_scene:
                    if t_iter >= self.cfg.scene.opt_start_iter:
                        scene_model = self.scene_gs.module if hasattr(self.scene_gs, 'module') else self.scene_gs
                        scene_model.optimizer.step()
                        scene_model.optimizer.zero_grad(set_to_none=True)
                t_iter += 1
        # Only rank 0 saves final checkpoint
        if self.cfg.rank == 0:
            self.save_ckpt()
        # Cleanup DDP
        if self.cfg.world_size > 1:
            dist.destroy_process_group()

    def save_ckpt(self, iter=None):
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        
        if self.human_gs:
            torch.save(self.human_gs.state_dict(), f'{self.cfg.logdir_ckpt}/human_{iter_s}.pth')
            
        if self.scene_gs:
            torch.save(self.scene_gs.state_dict(), f'{self.cfg.logdir_ckpt}/scene_{iter_s}.pth')
            self.scene_gs.save_ply(f'{self.cfg.logdir}/meshes/scene_{iter_s}_splat.ply')
            
        logger.info(f'Saved checkpoint {iter_s}')
                
    def scene_densification(self, visibility_filter, radii, viewspace_point_tensor, iteration):
        model = self.scene_gs.module if hasattr(self.scene_gs, 'module') else self.scene_gs
        
        model.max_radii2D[visibility_filter] = torch.max(
            model.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        model.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > self.cfg.scene.densify_from_iter and iteration % self.cfg.scene.densification_interval == 0:
            size_threshold = 20 if iteration > self.cfg.scene.opacity_reset_interval else None
            model.densify_and_prune(
                self.cfg.scene.densify_grad_threshold, 
                min_opacity=self.cfg.scene.prune_min_opacity, 
                extent=self.train_dataset.radius, 
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.scene.max_n_gaussians,
            )
        
        is_white = self.bg_color.sum().item() == 3.
        
        if iteration % self.cfg.scene.opacity_reset_interval == 0 or (is_white and iteration == self.cfg.scene.densify_from_iter):
            logger.info(f"[{iteration:06d}] Resetting opacity!!!")
            model.reset_opacity()
    
    def human_densification(self, human_gs_out, visibility_filter, radii, viewspace_point_tensor, iteration):
        model = self.human_gs.module if hasattr(self.human_gs, 'module') else self.human_gs
        
        model.max_radii2D[visibility_filter] = torch.max(
            model.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        model.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > self.cfg.human.densify_from_iter and iteration % self.cfg.human.densification_interval == 0:
            size_threshold = 20
            model.densify_and_prune(
                human_gs_out,
                self.cfg.human.densify_grad_threshold, 
                min_opacity=self.cfg.human.prune_min_opacity, 
                extent=self.cfg.human.densify_extent, 
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.human.max_n_gaussians,
            )
    
    @torch.no_grad()
    def validate(self, iter=None):
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
        
        if self.human_gs:
            self.human_gs.eval()
                
        methods = ['hugs', 'hugs_human']
        metrics = ['lpips', 'psnr', 'ssim']
        metrics = dict.fromkeys(['_'.join(x) for x in itertools.product(methods, metrics)])
        metrics = {k: [] for k in metrics}
        
        for idx, data in enumerate(tqdm(self.val_dataset, desc="Validation")):
            human_gs_out, scene_gs_out = None, None
            render_mode = self.cfg.mode
            
            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'], 
                    body_pose=data['body_pose'], 
                    betas=data['betas'], 
                    transl=data['transl'], 
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
                
            if self.scene_gs:
                if iter is not None:
                    if iter >= self.cfg.scene.opt_start_iter:
                        scene_gs_out = self.scene_gs.forward()
                    else:
                        render_mode = 'human'
                else:
                    scene_gs_out = self.scene_gs.forward()
                    
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=bg_color,
                render_mode=render_mode,
            )
            
            gt_image = data['rgb']
            
            image = render_pkg["render"]
            if self.cfg.dataset.name == 'zju':
                image = image * data['mask']
                gt_image = gt_image * data['mask']
            
            metrics['hugs_psnr'].append(psnr(image, gt_image).mean().double())
            metrics['hugs_ssim'].append(ssim(image, gt_image).mean().double())
            metrics['hugs_lpips'].append(self.lpips(image.clip(max=1), gt_image).mean().double())
            
            log_img = torchvision.utils.make_grid([gt_image, image], nrow=2, pad_value=1)
            imf = f'{self.cfg.logdir}/val/full_{iter_s}_{idx:03d}.png'
            os.makedirs(os.path.dirname(imf), exist_ok=True)
            torchvision.utils.save_image(log_img, imf)
            
            log_img = []
            if self.cfg.mode in ['human', 'human_scene']:
                bbox = data['bbox'].to(int)
                cropped_gt_image = gt_image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cropped_image = image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                log_img += [cropped_gt_image, cropped_image]
                
                metrics['hugs_human_psnr'].append(psnr(cropped_image, cropped_gt_image).mean().double())
                metrics['hugs_human_ssim'].append(ssim(cropped_image, cropped_gt_image).mean().double())
                metrics['hugs_human_lpips'].append(self.lpips(cropped_image.clip(max=1), cropped_gt_image).mean().double())
            
            if len(log_img) > 0:
                log_img = torchvision.utils.make_grid(log_img, nrow=len(log_img), pad_value=1)
                torchvision.utils.save_image(log_img, f'{self.cfg.logdir}/val/human_{iter_s}_{idx:03d}.png')
        
        
        self.eval_metrics[iter_s] = {}
        
        for k, v in metrics.items():
            if v == []:
                continue
            
            logger.info(f"{iter_s} - {k.upper()}: {torch.stack(v).mean().item():.4f}")
            self.eval_metrics[iter_s][k] = torch.stack(v).mean().item()
        
        torch.save(metrics, f'{self.cfg.logdir}/val/eval_{iter_s}.pth')
    
    @torch.no_grad()
    def animate(self, iter=None, keep_images=False):
        if self.anim_dataset is None:
            logger.info("No animation dataset found")
            return 0
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        if self.human_gs:
            self.human_gs.eval()
        
        os.makedirs(f'{self.cfg.logdir}/anim/', exist_ok=True)
        
        for idx, data in enumerate(tqdm(self.anim_dataset, desc="Animation")):
            human_gs_out, scene_gs_out = None, None
            
            if self.human_gs:
                ext_tfs = (data['manual_trans'], data['manual_rotmat'], data['manual_scale'])
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=ext_tfs,
                )
            
            if self.scene_gs:
                scene_gs_out = self.scene_gs.forward()
                    
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode=self.cfg.mode,
            )
            
            image = render_pkg["render"]
            
            torchvision.utils.save_image(image, f'{self.cfg.logdir}/anim/{idx:05d}.png')
            
        video_fname = f'{self.cfg.logdir}/anim_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/anim/', video_fname, fps=20)
        if not keep_images:
            # shutil.rmtree(f'{self.cfg.logdir}/anim/')
            os.makedirs(f'{self.cfg.logdir}/anim/', exist_ok=True)
    
    @torch.no_grad()
    def render_canonical(self, iter=None, nframes=100, is_train_progress=False, pose_type=None):
        iter_s = 'final' if iter is None else f'{iter:06d}'
        iter_s += f'_{pose_type}' if pose_type is not None else ''
        
        if self.human_gs:
            self.human_gs.eval()
        
        os.makedirs(f'{self.cfg.logdir}/canon/', exist_ok=True)
        
        camera_params = get_rotating_camera(
            dist=5.0, img_size=256 if is_train_progress else 512, 
            nframes=nframes, device='cuda',
            angle_limit=torch.pi if is_train_progress else 2*torch.pi,
        )
        
        betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.train_dataset.smpl_params['betas'][0]
        
        static_smpl_params = get_smpl_static_params(
            betas=betas,
            pose_type=self.cfg.human.canon_pose_type if pose_type is None else pose_type,
        )
        
        if is_train_progress:
            progress_imgs = []
        
        pbar = range(nframes) if is_train_progress else tqdm(range(nframes), desc="Canonical:")
        
        for idx in pbar:
            human_gs_out, scene_gs_out = None, None
            
            cam_p = camera_params[idx]
            data = dict(static_smpl_params, **cam_p)

            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
                
            if is_train_progress:
                scale_mod = 0.5
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                    scaling_modifier=scale_mod,
                )
                
                image = render_pkg["render"]
                
                progress_imgs.append(image)
                
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                )
                
                image = render_pkg["render"]
                
                progress_imgs.append(image)
                
            else:
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                )
                
                image = render_pkg["render"]
                
                torchvision.utils.save_image(image, f'{self.cfg.logdir}/canon/{idx:05d}.png')
        
        if is_train_progress:
            os.makedirs(f'{self.cfg.logdir}/train_progress/', exist_ok=True)
            log_img = torchvision.utils.make_grid(progress_imgs, nrow=4, pad_value=0)
            save_image(log_img, f'{self.cfg.logdir}/train_progress/{iter:06d}.png', 
                       text_labels=f"{iter:06d}, n_gs={self.human_gs.n_gs}")
            return
        
        video_fname = f'{self.cfg.logdir}/canon_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/canon/', video_fname, fps=10)
        # shutil.rmtree(f'{self.cfg.logdir}/canon/')
        os.makedirs(f'{self.cfg.logdir}/canon/', exist_ok=True)
        
    def render_poses(self, camera_params, smpl_params, pose_type='a_pose', bg_color='white'):
    
        if self.human_gs:
            self.human_gs.eval()
        
        betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.val_dataset.smpl_params['betas'][0]
        
        nframes = len(camera_params)
        
        canon_forward_out = None
        if hasattr(self.human_gs, 'canon_forward'):
            canon_forward_out = self.human_gs.canon_forward()
        
        pbar = tqdm(range(nframes), desc="Canonical:")
        if bg_color == 'white':
            bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color == 'black':
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            
            
        imgs = []
        for idx in pbar:
            human_gs_out, scene_gs_out = None, None
            
            cam_p = camera_params[idx]
            data = dict(smpl_params, **cam_p)

            if self.human_gs:
                if canon_forward_out is not None:
                    human_gs_out = self.human_gs.forward_test(
                        canon_forward_out,
                        global_orient=data['global_orient'],
                        body_pose=data['body_pose'],
                        betas=data['betas'],
                        transl=data['transl'],
                        smpl_scale=data['smpl_scale'],
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                    )
                else:
                    human_gs_out = self.human_gs.forward(
                        global_orient=data['global_orient'],
                        body_pose=data['body_pose'],
                        betas=data['betas'],
                        transl=data['transl'],
                        smpl_scale=data['smpl_scale'],
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                    )

            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode='human',
            )
            image = render_pkg["render"]
            imgs.append(image)
        return imgs