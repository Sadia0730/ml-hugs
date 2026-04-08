"""
Temporal Consistency Metric for Cloth-HUGS
===========================================
Computes the temporal Optical Flow error (tOF) between consecutive rendered frames.

Method (standard in dynamic NeRF / Gaussian Splatting papers):
  1. Render frames {I_t} and {I_{t+1}} for consecutive poses.
  2. Estimate dense optical flow F_{t→t+1} from the *ground-truth* video using
     OpenCV Farneback.
  3. Warp the rendered frame I_t to t+1 using F: W(I_t, F).
  4. tOF error = ||W(I_t, F) - I_{t+1}||_1  (mean L1 over valid pixels).

A lower tOF indicates better temporal consistency.

Usage
-----
python temporal_consistency.py \
    --cfg  output/human_scene/cloth_zju_386_train/2025-11-01_02-38-10/config_train.yaml \
    --ckpt output/human_scene/cloth_zju_386_train/2025-11-01_02-38-10/ckpt/human_final.pth \
    --n_pairs 50
"""

import os
import sys
import json
import math
import argparse

import cv2
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
def warp_image(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp img with dense flow field.
    img  : H×W×3 float32 in [0,1]
    flow : H×W×2  (dx, dy) in pixels
    Returns warped image same shape.
    """
    h, w = img.shape[:2]
    # Build absolute coordinate map
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def tof_error(warped: np.ndarray, target: np.ndarray, mask=None) -> float:
    """
    Mean L1 error between warped I_t and actual I_{t+1}.
    mask: optional binary mask (H×W) for human region only.
    """
    diff = np.abs(warped.astype(np.float32) - target.astype(np.float32))
    if mask is not None:
        diff = diff[mask > 0]
    return float(diff.mean())


# ─────────────────────────────────────────────────────────────────────────────
def build_model(cfg, ckpt_path):
    from hugs.models.hugs_trimlp import HUGS_TRIMLP
    from hugs.datasets.zju import ZJUMoCapDataset
    from hugs.datasets import NeumanDataset

    if cfg.dataset.name == 'zju':
        dataset = ZJUMoCapDataset(
            cfg.dataset.seq, split='val',
            render_mode=getattr(cfg, 'mode', 'human'),
            cloth_upper=getattr(cfg.dataset, 'cloth_upper', 'tshirt'),
            cloth_lower=getattr(cfg.dataset, 'cloth_lower', 'pants'),
            cloth_dir=getattr(cfg.dataset, 'cloth_dir', 'assets/snug'),
            dataset_path=getattr(cfg.dataset, 'dataset_path', 'data/zju_mocap/processed'),
        )
    else:
        dataset = NeumanDataset(
            cfg.dataset.seq, 'val',
            render_mode=getattr(cfg, 'mode', 'human'),
            cloth_upper=getattr(cfg.dataset, 'cloth_upper', 'tshirt'),
            cloth_lower=getattr(cfg.dataset, 'cloth_lower', 'pants'),
            cloth_dir=getattr(cfg.dataset, 'cloth_dir', 'assets/snug'),
            dataset_path=getattr(cfg.dataset, 'dataset_path', None),
        )

    init_betas = torch.stack([x['betas'] for x in dataset.cached_data], dim=0)
    human_gs = HUGS_TRIMLP(
        sh_degree=cfg.human.sh_degree,
        n_subdivision=cfg.human.n_subdivision,
        use_surface=cfg.human.use_surface,
        init_2d=cfg.human.init_2d,
        rotate_sh=cfg.human.rotate_sh,
        isotropic=cfg.human.isotropic,
        init_scale_multiplier=cfg.human.init_scale_multiplier,
        n_features=32,
        use_deformer=cfg.human.use_deformer,
        disable_posedirs=cfg.human.disable_posedirs,
        triplane_res=cfg.human.triplane_res,
        betas=init_betas[0],
    )
    human_gs.create_betas(init_betas[0], False)

    if hasattr(dataset, 'cloth_vertices') and dataset.cloth_vertices is not None:
        human_gs.initialize_cloth(dataset.cloth_vertices, dataset.cloth_faces)

    ckpt = torch.load(ckpt_path, map_location='cuda')
    human_gs.load_state_dict(ckpt)
    human_gs.eval()
    return human_gs, dataset


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run(cfg_path, ckpt_path, n_pairs=50, output_json=None):
    cfg = OmegaConf.load(cfg_path)
    logger.info("Building model …")
    human_gs, dataset = build_model(cfg, ckpt_path)

    from hugs.renderer.gs_renderer import render_human_scene, render

    bg_color = torch.zeros(3, dtype=torch.float32, device='cuda')
    n_frames = min(len(dataset), n_pairs + 1)

    logger.info(f"Rendering {n_frames} consecutive frames …")
    rendered = []  # list of H×W×3 numpy uint8
    gt_frames = []
    masks = []

    for idx in range(n_frames):
        data = dataset[idx]
        # Move to CUDA
        data_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        smpl_scale = data_cuda.get('smpl_scale', torch.ones(1, device='cuda'))
        if smpl_scale.dim() == 0:
            smpl_scale = smpl_scale.unsqueeze(0)

        human_pack = human_gs.forward(
            global_orient=data_cuda.get('global_orient'),
            body_pose=data_cuda.get('body_pose'),
            betas=data_cuda.get('betas'),
            transl=data_cuda.get('transl'),
            smpl_scale=smpl_scale,
            dataset_idx=-1,
            is_train=False,
        )

        if isinstance(human_pack, dict):
            body_out = human_pack['body']
            cloth_out = human_pack.get('cloth', None)
        else:
            body_out = human_pack
            cloth_out = None

        render_pkg = render(
            means3D=body_out['xyz'],
            feats=body_out['shs'],
            opacity=body_out['opacity'],
            scales=body_out['scales'],
            rotations=body_out['rotq'],
            data=data_cuda,
            bg_color=bg_color,
            active_sh_degree=body_out['active_sh_degree'],
        )
        img_tensor = render_pkg['render']  # 3×H×W in [0,1]
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rendered.append(img_np)

        gt_tensor = data_cuda.get('rgb')  # 3×H×W
        if gt_tensor is not None:
            gt_np = (gt_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_frames.append(gt_np)
        else:
            gt_frames.append(img_np.copy())

        # Optional human mask for masked tOF
        if 'mask' in data_cuda:
            mask = data_cuda['mask'].cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]  # take first channel
            masks.append((mask > 0.5).astype(np.uint8))
        else:
            masks.append(None)

    logger.info(f"Computing tOF for {n_pairs} consecutive pairs …")
    tof_vals_full   = []
    tof_vals_masked = []
    warp_err_gt     = []  # GT-flow warping error (using GT frames for flow)

    for i in range(n_pairs):
        frame_t  = rendered[i]        # I_t  rendered
        frame_t1 = rendered[i + 1]    # I_{t+1} rendered

        gt_t   = gt_frames[i]
        gt_t1  = gt_frames[i + 1]

        # ── Compute optical flow on GT frames (more stable signal) ────────────
        gt_gray_t  = cv2.cvtColor(gt_t,  cv2.COLOR_RGB2GRAY)
        gt_gray_t1 = cv2.cvtColor(gt_t1, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gt_gray_t, gt_gray_t1,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )  # H×W×2

        # ── Warp rendered I_t → t+1 ───────────────────────────────────────────
        frame_t_f32  = frame_t.astype(np.float32)
        frame_t1_f32 = frame_t1.astype(np.float32)

        warped = warp_image(frame_t_f32, flow)

        # Full-frame tOF (normalized to [0,1])
        err_full = tof_error(warped / 255.0, frame_t1_f32 / 255.0)
        tof_vals_full.append(err_full)

        # Masked tOF (human region only)
        mask = masks[i + 1]
        if mask is not None:
            err_masked = tof_error(warped / 255.0, frame_t1_f32 / 255.0, mask)
            tof_vals_masked.append(err_masked)

        # ── GT-flow warping error (sanity / upper bound reference) ───────────
        gt_t_f32  = gt_t.astype(np.float32)
        gt_t1_f32 = gt_t1.astype(np.float32)
        warped_gt = warp_image(gt_t_f32, flow)
        warp_err_gt.append(tof_error(warped_gt / 255.0, gt_t1_f32 / 255.0))

    # ── Report ────────────────────────────────────────────────────────────────
    tof_mean   = float(np.mean(tof_vals_full))
    tof_std    = float(np.std(tof_vals_full))
    gt_warp_mean = float(np.mean(warp_err_gt))

    sep = "─" * 56
    print(f"\n{sep}")
    print("  Temporal Consistency (tOF) — Cloth-HUGS")
    print(sep)
    print(f"  Sequence         : {cfg.dataset.name}/{cfg.dataset.seq}")
    print(f"  Consecutive pairs: {n_pairs}")
    print(sep)
    print(f"  tOF (full frame) : {tof_mean:.4f} ± {tof_std:.4f}")
    if tof_vals_masked:
        m_mean = float(np.mean(tof_vals_masked))
        m_std  = float(np.std(tof_vals_masked))
        print(f"  tOF (human mask) : {m_mean:.4f} ± {m_std:.4f}")
    print(f"  GT warp baseline : {gt_warp_mean:.4f}  (lower bound — flow estimation error)")
    print(sep)
    print("  Lower tOF = better temporal consistency.")
    print()

    results = {
        "sequence"         : f"{cfg.dataset.name}/{cfg.dataset.seq}",
        "n_pairs"          : n_pairs,
        "tof_full_mean"    : tof_mean,
        "tof_full_std"     : tof_std,
        "gt_warp_baseline" : gt_warp_mean,
    }
    if tof_vals_masked:
        results["tof_masked_mean"] = float(np.mean(tof_vals_masked))
        results["tof_masked_std"]  = float(np.std(tof_vals_masked))

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_json}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",  required=True, help="config_train.yaml path")
    parser.add_argument("--ckpt", required=True, help="human_final.pth path")
    parser.add_argument("--n_pairs", type=int, default=50, help="Consecutive frame pairs to evaluate")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    run(args.cfg, args.ckpt, args.n_pairs, args.output_json)
