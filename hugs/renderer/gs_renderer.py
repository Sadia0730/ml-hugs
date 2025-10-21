# Code adapted from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, 
    GaussianRasterizer
)

from hugs.utils.spherical_harmonics import SH2RGB
from hugs.utils.rotations import quaternion_to_matrix


def render_human_scene(
    data, 
    human_gs_out,        # body (or dict with body+cloth if you pass differently – see note)
    scene_gs_out,
    bg_color, 
    human_bg_color=None,
    scaling_modifier=1.0, 
    render_mode='human_scene',
    render_human_separate=False,
    # >>> cloth add
    cloth_gs_out=None,   # pass HUGS_TRIMLP.forward(... )["cloth"] here if available
):
    """
    If cloth_gs_out is provided and render_mode is 'human_scene', we do:
      base = render(body+scene)   # depth blending
      cloth_rgb, cloth_vis = separate passes
      final = cloth_rgb * cloth_vis + base * (1 - cloth_vis)
    Otherwise behaves exactly like before.
    """

    # --- original body/scene switch ---
    feats = None
    if render_mode == 'human_scene':
        feats = torch.cat([human_gs_out['shs'], scene_gs_out['shs']], dim=0)
        means3D = torch.cat([human_gs_out['xyz'], scene_gs_out['xyz']], dim=0)
        opacity = torch.cat([human_gs_out['opacity'], scene_gs_out['opacity']], dim=0)
        scales = torch.cat([human_gs_out['scales'], scene_gs_out['scales']], dim=0)
        rotations = torch.cat([human_gs_out['rotq'], scene_gs_out['rotq']], dim=0)
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'human':
        feats = human_gs_out['shs']
        means3D = human_gs_out['xyz']
        opacity = human_gs_out['opacity']
        scales = human_gs_out['scales']
        rotations = human_gs_out['rotq']
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'scene':
        feats = scene_gs_out['shs']
        means3D = scene_gs_out['xyz']
        opacity = scene_gs_out['opacity']
        scales = scene_gs_out['scales']
        rotations = scene_gs_out['rotq']
        active_sh_degree = scene_gs_out['active_sh_degree']
    else:
        raise ValueError(f'Unknown render mode: {render_mode}')
    
    # --- base render (unchanged path) ---
    render_pkg = render(
        means3D=means3D,
        feats=feats,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        data=data,
        scaling_modifier=scaling_modifier,
        bg_color=bg_color,
        active_sh_degree=active_sh_degree,
    )

    # --- prepare extras from original code (unchanged) ---
    if render_human_separate and render_mode == 'human_scene':
        render_human_pkg = render(
            means3D=human_gs_out['xyz'],
            feats=human_gs_out['shs'],
            opacity=human_gs_out['opacity'],
            scales=human_gs_out['scales'],
            rotations=human_gs_out['rotq'],
            data=data,
            scaling_modifier=scaling_modifier,
            bg_color=human_bg_color if human_bg_color is not None else bg_color,
            active_sh_degree=human_gs_out['active_sh_degree'],
        )
        render_pkg['human_img'] = render_human_pkg['render']
        render_pkg['human_visibility_filter'] = render_human_pkg['visibility_filter']
        render_pkg['human_radii'] = render_human_pkg['radii']
        
    if render_mode == 'human':
        render_pkg['human_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['human_radii'] = render_pkg['radii']
    elif render_mode == 'human_scene':
        human_n_gs = human_gs_out['xyz'].shape[0]
        scene_n_gs = scene_gs_out['xyz'].shape[0]
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter'][human_n_gs:]
        render_pkg['scene_radii'] = render_pkg['radii'][human_n_gs:]
        if not 'human_visibility_filter' in render_pkg.keys():
            render_pkg['human_visibility_filter'] = render_pkg['visibility_filter'][:-scene_n_gs]
            render_pkg['human_radii'] = render_pkg['radii'][:-scene_n_gs]
            
    elif render_mode == 'scene':
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['scene_radii'] = render_pkg['radii']

    # >>> cloth add: do the 2nd pass and composite
    if cloth_gs_out is not None and render_mode == 'human_scene':
        # 1) base image (body+scene) already computed:
        base_rgb = render_pkg["render"]

        # 2) cloth color (with SH) - track visibility/radii for densification
        cloth_rgb, cloth_render_info = _render_colors_only(
            means3D=cloth_gs_out["xyz"],
            feats=cloth_gs_out["shs"],
            opacity=cloth_gs_out["opacity"],
            scales=cloth_gs_out["scales"],
            rotations=cloth_gs_out["rotq"],
            data=data,
            scaling_modifier=scaling_modifier,
            bg_color=torch.zeros(3, device=base_rgb.device),
            sh_degree=cloth_gs_out["active_sh_degree"],
        )
        
        # Store cloth-specific rendering info for densification
        render_pkg["cloth_visibility_filter"] = cloth_render_info["visibility_filter"]
        render_pkg["cloth_radii"] = cloth_render_info["radii"]
        render_pkg["cloth_viewspace_points"] = cloth_render_info["viewspace_points"]

        # 3) cloth visibility vs (body+scene) in one pass (depth-aware matte)
        blockers = {
            "xyz":   torch.cat([human_gs_out["xyz"], scene_gs_out["xyz"]], dim=0),
            "opacity":torch.cat([human_gs_out["opacity"], scene_gs_out["opacity"]], dim=0),
            "scales": torch.cat([human_gs_out["scales"], scene_gs_out["scales"]], dim=0),
            "rotq":   torch.cat([human_gs_out["rotq"],   scene_gs_out["rotq"]],   dim=0),
        }
        cloth_vis = _render_visibility_matte(
            cloth=cloth_gs_out, blockers=blockers,
            data=data, scaling_modifier=scaling_modifier, sh_degree=0
        )  # HxWx1

        # 4) composite (cloth OVER base)
        comp = cloth_rgb * cloth_vis + base_rgb * (1.0 - cloth_vis)
        render_pkg["render"] = torch.clamp(comp, 0.0, 1.0)
        render_pkg["cloth_rgb"] = cloth_rgb
        render_pkg["cloth_vis"] = cloth_vis

    return render_pkg

    
def render(means3D, feats, opacity, scales, rotations, data, scaling_modifier=1.0, bg_color=None, active_sh_degree=0):
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
        
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        pass

    means2D = screenspace_points

    # Set up rasterization configuration
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)

    shs, rgb = None, None
    if len(feats.shape) == 2:
        rgb = feats
    else:
        shs = feats

    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=active_sh_degree,
        campos=data['camera_center'],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        colors_precomp=rgb,
    )
    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }
def _render_colors_only(means3D, feats, opacity, scales, rotations, data, scaling_modifier, bg_color, sh_degree):
    """One standard render that returns RGB only."""
    out = render(
        means3D=means3D, feats=feats, opacity=opacity, scales=scales, rotations=rotations,
        data=data, scaling_modifier=scaling_modifier, bg_color=bg_color, active_sh_degree=sh_degree
    )
    return out["render"], out

def _render_alpha_only(means3D, opacity, scales, rotations, data, scaling_modifier, sh_degree):
    """
    Render with colors_precomp=1 so the output equals the accumulated alpha (coverage).
    Works with the 3DGS rasterizer because it alpha-composites linearly.
    """
    # Make a dummy "colors_precomp" tensor of ones (RGB)
    ones = torch.ones((means3D.shape[0], 3), dtype=means3D.dtype, device=means3D.device)
    out = render(
        means3D=means3D, feats=ones, opacity=opacity, scales=scales, rotations=rotations,
        data=data, scaling_modifier=scaling_modifier, bg_color=torch.zeros(3, device=means3D.device),
        active_sh_degree=0  # colors_precomp path ignores SH anyway
    )
    # single-channel alpha (use any channel; they’re identical)
    alpha = out["render"][..., :1]
    return alpha

def _render_visibility_matte(cloth, blockers, data, scaling_modifier, sh_degree):
    """
    Depth-aware visibility of *cloth* in front of (body+scene):
    - cloth gaussians colored to 1
    - blockers (body+scene) colored to 0
    The rasterizer’s alpha compositing + depth resolves which layer is visible.
    """
    # concatenate in the order [cloth, blockers] so both sets are in the same pass
    means3D   = torch.cat([cloth["xyz"], blockers["xyz"]], dim=0)
    opacity   = torch.cat([cloth["opacity"], blockers["opacity"]], dim=0)
    scales    = torch.cat([cloth["scales"], blockers["scales"]], dim=0)
    rotations = torch.cat([cloth["rotq"],  blockers["rotq"]],  dim=0)

    # colors: cloth -> ones, blockers -> zeros
    cloth_ones   = torch.ones((cloth["xyz"].shape[0], 3), dtype=means3D.dtype, device=means3D.device)
    blockers_zero= torch.zeros((blockers["xyz"].shape[0], 3), dtype=means3D.dtype, device=means3D.device)
    feats = torch.cat([cloth_ones, blockers_zero], dim=0)  # use colors_precomp route

    out = render(
        means3D=means3D, feats=feats, opacity=opacity, scales=scales, rotations=rotations,
        data=data, scaling_modifier=scaling_modifier,
        bg_color=torch.zeros(3, device=means3D.device), active_sh_degree=0
    )
    # matte of visible cloth after depth with blockers
    cloth_vis = out["render"][..., :1]
    return cloth_vis
