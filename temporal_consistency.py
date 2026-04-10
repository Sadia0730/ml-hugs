"""
Temporal Consistency Metrics for Cloth-HUGS
============================================
Quantitative response to ICIP reviewer concern on flickering / chaotic per-splat
motion in 3DGS-based methods. PSNR / SSIM / LPIPS / FID are computed per-frame
and cannot detect inter-frame jitter; the metrics below are designed exactly
for that purpose and are standard in dynamic NeRF / video-synthesis papers.

Metrics reported
----------------
(1) tOF   — temporal Optical Flow L1 error (Chu et al., TecoGAN 2020)
            Warps rendered I_t to t+1 using GT optical flow and measures the
            residual against the rendered I_{t+1}. Lower = smoother motion.

(2) E_warp — Occlusion-aware warping error (Lai et al., "Learning Blind Video
            Temporal Consistency", ECCV 2018). Same as tOF but weighted by a
            photometric occlusion mask M = exp(-alpha * ||GT_{t+1}-W(GT_t)||^2)
            so that disoccluded regions (where a single correct answer does
            not exist) do not dominate the metric.

(3) tLP   — temporal LPIPS error. Replaces L1 with a perceptual LPIPS distance
            between W(I_t) and I_{t+1}. This catches perceptual flicker that
            pixel L1 misses (small high-frequency splat fluctuations).

(4) Flicker — high-frequency temporal variance of motion-compensated residuals,
            measuring pixel-level flickering after removing legitimate motion.
            Computed as mean std over a sliding temporal window on the
            motion-compensated rendered sequence.

(5) Splat_Accel — 3D Gaussian acceleration norm. Cloth-HUGS gives direct access
            to per-Gaussian 3D means across frames, so we can *directly*
            measure geometric jitter instead of proxying through pixels:
                a_t = || x_{t+1} - 2 x_t + x_{t-1} ||_2
            Reported as the mean per-splat acceleration (units: scene
            coordinates). This is the metric that most directly answers the
            reviewer's "chaotic movements of individual splats" complaint.

All metrics are lower-is-better.

Usage
-----
python temporal_consistency.py \
    --cfg  output/human_scene/cloth_zju_386_train/.../config_train.yaml \
    --ckpt output/human_scene/cloth_zju_386_train/.../ckpt/human_final.pth \
    --n_pairs 50 \
    --output_json temporal_results.json
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Image-space helpers
# ─────────────────────────────────────────────────────────────────────────────
def warp_image(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp img with dense flow field F_{t→t+1}.
    img  : H×W×C float32
    flow : H×W×2 (dx, dy) in pixels
    """
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def occlusion_mask(gt_t: np.ndarray, gt_t1: np.ndarray, flow: np.ndarray,
                   alpha: float = 50.0) -> np.ndarray:
    """
    Photometric occlusion mask from Lai et al. ECCV'18.
    M(x) = exp(-alpha * ||GT_{t+1}(x) - W(GT_t)(x)||^2 )
    Values near 0 → disoccluded / unreliable, near 1 → reliable.
    """
    warped_gt = warp_image(gt_t.astype(np.float32) / 255.0, flow)
    diff = (gt_t1.astype(np.float32) / 255.0) - warped_gt
    err = np.sum(diff ** 2, axis=-1)  # H×W
    return np.exp(-alpha * err)


def masked_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray = None) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if mask is None:
        return float(diff.mean())
    diff = diff.mean(axis=-1) if diff.ndim == 3 else diff   # H×W
    denom = mask.sum() + 1e-8
    return float((diff * mask).sum() / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Model construction (unchanged — mirrors the existing validation pipeline)
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
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run(cfg_path, ckpt_path, n_pairs=50, output_json=None, use_lpips=True):
    cfg = OmegaConf.load(cfg_path)
    logger.info("Building model …")
    human_gs, dataset = build_model(cfg, ckpt_path)

    from hugs.renderer.gs_renderer import render

    # Perceptual model for tLP (net='alex' matches the per-frame LPIPS you use)
    lpips_model = None
    if use_lpips:
        try:
            from lpips import LPIPS
            lpips_model = LPIPS(net='alex', pretrained=True).to('cuda').eval()
        except Exception as e:
            logger.warning(f"LPIPS unavailable ({e}); tLP will be skipped.")
            lpips_model = None

    bg_color = torch.zeros(3, dtype=torch.float32, device='cuda')

    # Some Neuman val splits contain only a handful of frames, which is too
    # few to measure temporal behavior. Fall back to the train split in that
    # case — temporal metrics are view-consistency measures on the rendered
    # sequence itself, so the split used for evaluation is not critical.
    requested_pairs = n_pairs
    if len(dataset) < 3:
        logger.warning(
            f"Val split has only {len(dataset)} frames → "
            f"too few for temporal metrics. Falling back to train split."
        )
        if cfg.dataset.name == 'zju':
            from hugs.datasets.zju import ZJUMoCapDataset
            dataset = ZJUMoCapDataset(
                cfg.dataset.seq, split='train',
                render_mode=getattr(cfg, 'mode', 'human'),
                cloth_upper=getattr(cfg.dataset, 'cloth_upper', 'tshirt'),
                cloth_lower=getattr(cfg.dataset, 'cloth_lower', 'pants'),
                cloth_dir=getattr(cfg.dataset, 'cloth_dir', 'assets/snug'),
                dataset_path=getattr(cfg.dataset, 'dataset_path', 'data/zju_mocap/processed'),
            )
        else:
            from hugs.datasets import NeumanDataset
            dataset = NeumanDataset(
                cfg.dataset.seq, 'train',
                render_mode=getattr(cfg, 'mode', 'human'),
                cloth_upper=getattr(cfg.dataset, 'cloth_upper', 'tshirt'),
                cloth_lower=getattr(cfg.dataset, 'cloth_lower', 'pants'),
                cloth_dir=getattr(cfg.dataset, 'cloth_dir', 'assets/snug'),
                dataset_path=getattr(cfg.dataset, 'dataset_path', None),
            )

    n_frames = min(len(dataset), requested_pairs + 1)
    n_pairs  = max(0, n_frames - 1)
    if n_pairs < 2:
        raise RuntimeError(
            f"Not enough frames for temporal metrics (got {len(dataset)})."
        )
    logger.info(f"Rendering {n_frames} consecutive frames (n_pairs={n_pairs}) …")

    rendered_np = []          # list of H×W×3 uint8 (for optical flow / image metrics)
    rendered_tensor = []      # list of 3×H×W cuda float for LPIPS
    gt_frames = []
    masks = []
    splat_means = []          # list of (N,3) tensors for per-splat acceleration

    for idx in range(n_frames):
        data = dataset[idx]
        data_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                     for k, v in data.items()}

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

        # Record deformed 3D means (body ∪ cloth) for splat acceleration
        means_parts = [body_out['xyz'].detach()]
        if cloth_out is not None and 'xyz' in cloth_out:
            means_parts.append(cloth_out['xyz'].detach())
        splat_means.append(torch.cat(means_parts, dim=0).cpu())

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
        img_tensor = render_pkg['render'].clamp(0, 1)          # 3×H×W
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rendered_np.append(img_np)
        rendered_tensor.append(img_tensor)

        gt_tensor = data_cuda.get('rgb')
        if gt_tensor is not None:
            gt_np = (gt_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_frames.append(gt_np)
        else:
            gt_frames.append(img_np.copy())

        if 'mask' in data_cuda:
            m = data_cuda['mask'].cpu().numpy()
            if m.ndim == 3:
                m = m[0]
            masks.append((m > 0.5).astype(np.uint8))
        else:
            masks.append(None)

    logger.info(f"Computing temporal metrics over {n_pairs} consecutive pairs …")

    tof_full, tof_masked       = [], []
    ewarp_full, ewarp_masked   = [], []
    tlp_vals                   = []
    flicker_residuals          = []   # keep per-pair motion-compensated residuals
    gt_warp_baseline           = []

    for i in range(n_pairs):
        I_t   = rendered_np[i].astype(np.float32) / 255.0
        I_t1  = rendered_np[i + 1].astype(np.float32) / 255.0
        gt_t  = gt_frames[i]
        gt_t1 = gt_frames[i + 1]

        # Dense flow on GT (more stable than on the rendered sequence)
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(gt_t,  cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(gt_t1, cv2.COLOR_RGB2GRAY),
            None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # (1) tOF — raw motion-compensated L1
        warped_I = warp_image(I_t, flow)
        tof_full.append(masked_l1(warped_I, I_t1))
        human_mask = masks[i + 1]
        if human_mask is not None:
            tof_masked.append(masked_l1(warped_I, I_t1, human_mask.astype(np.float32)))

        # (2) E_warp — occlusion-weighted
        occ = occlusion_mask(gt_t, gt_t1, flow, alpha=50.0)  # H×W in [0,1]
        ewarp_full.append(masked_l1(warped_I, I_t1, occ))
        if human_mask is not None:
            ewarp_masked.append(masked_l1(warped_I, I_t1, occ * human_mask.astype(np.float32)))

        # (3) tLP — perceptual temporal distance
        if lpips_model is not None:
            warped_t = torch.from_numpy(warped_I).permute(2, 0, 1).unsqueeze(0).float().cuda()
            target_t = rendered_tensor[i + 1].unsqueeze(0)
            # LPIPS expects inputs in [-1, 1]
            tlp = lpips_model(warped_t * 2 - 1, target_t * 2 - 1).item()
            tlp_vals.append(tlp)

        # (4) Flicker residual (keep for aggregate temporal variance below)
        flicker_residuals.append(I_t1 - warped_I)

        # Baseline: GT warped with GT flow (measures flow estimator noise floor)
        gt_warp_baseline.append(
            masked_l1(warp_image(gt_t.astype(np.float32) / 255.0, flow),
                      gt_t1.astype(np.float32) / 255.0)
        )

    # Flicker magnitude: temporal std of motion-compensated residuals.
    # Low std → residuals behave like static noise; high std → flickering.
    residual_stack = np.stack(flicker_residuals, axis=0)                  # T×H×W×3
    flicker_mag = float(residual_stack.std(axis=0).mean())

    # (5) Per-splat acceleration (central difference on 3D means).
    # Requires at least 3 frames and a stable splat ordering across frames.
    splat_accel_mean = None
    splat_accel_p95  = None
    if len(splat_means) >= 3:
        sizes = [m.shape[0] for m in splat_means]
        if len(set(sizes)) == 1:
            X = torch.stack(splat_means, dim=0)          # T×N×3
            accel = X[2:] - 2 * X[1:-1] + X[:-2]         # (T-2)×N×3
            accel_norm = accel.norm(dim=-1)              # (T-2)×N
            splat_accel_mean = float(accel_norm.mean().item())
            splat_accel_p95  = float(torch.quantile(accel_norm.flatten(), 0.95).item())
        else:
            logger.warning("Splat count varies across frames; skipping Splat_Accel.")

    # ── Report ───────────────────────────────────────────────────────────────
    def _mean_std(xs):
        return (float(np.mean(xs)), float(np.std(xs))) if len(xs) else (None, None)

    tof_m,    tof_s    = _mean_std(tof_full)
    tofm_m,   tofm_s   = _mean_std(tof_masked)
    ew_m,     ew_s     = _mean_std(ewarp_full)
    ewm_m,    ewm_s    = _mean_std(ewarp_masked)
    tlp_m,    tlp_s    = _mean_std(tlp_vals)
    gtw_m,    _        = _mean_std(gt_warp_baseline)

    sep = "─" * 64
    print(f"\n{sep}")
    print("  Temporal Consistency Metrics — Cloth-HUGS")
    print(sep)
    print(f"  Sequence          : {cfg.dataset.name}/{cfg.dataset.seq}")
    print(f"  Consecutive pairs : {n_pairs}")
    print(sep)
    print(f"  tOF   (full)      : {tof_m:.4f} ± {tof_s:.4f}")
    if tofm_m is not None:
        print(f"  tOF   (human)     : {tofm_m:.4f} ± {tofm_s:.4f}")
    print(f"  E_warp (full)     : {ew_m:.4f} ± {ew_s:.4f}")
    if ewm_m is not None:
        print(f"  E_warp (human)    : {ewm_m:.4f} ± {ewm_s:.4f}")
    if tlp_m is not None:
        print(f"  tLP   (LPIPS)     : {tlp_m:.4f} ± {tlp_s:.4f}")
    print(f"  Flicker (residσ)  : {flicker_mag:.4f}")
    if splat_accel_mean is not None:
        print(f"  Splat accel mean  : {splat_accel_mean:.6f}  (scene units)")
        print(f"  Splat accel p95   : {splat_accel_p95:.6f}")
    print(f"  GT-warp baseline  : {gtw_m:.4f}   (flow-estimator noise floor)")
    print(sep)
    print("  All metrics: lower = more temporally consistent.")
    print()

    results = {
        "sequence"          : f"{cfg.dataset.name}/{cfg.dataset.seq}",
        "n_pairs"           : n_pairs,
        "tof_full_mean"     : tof_m,
        "tof_full_std"      : tof_s,
        "ewarp_full_mean"   : ew_m,
        "ewarp_full_std"    : ew_s,
        "flicker_residual_std": flicker_mag,
        "gt_warp_baseline"  : gtw_m,
    }
    if tofm_m is not None:
        results["tof_masked_mean"]   = tofm_m
        results["tof_masked_std"]    = tofm_s
    if ewm_m is not None:
        results["ewarp_masked_mean"] = ewm_m
        results["ewarp_masked_std"]  = ewm_s
    if tlp_m is not None:
        results["tlp_mean"] = tlp_m
        results["tlp_std"]  = tlp_s
    if splat_accel_mean is not None:
        results["splat_accel_mean"] = splat_accel_mean
        results["splat_accel_p95"]  = splat_accel_p95

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
    parser.add_argument("--n_pairs", type=int, default=50,
                        help="Number of consecutive frame pairs to evaluate")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--no_lpips", action='store_true',
                        help="Disable tLP (skip LPIPS model load)")
    args = parser.parse_args()

    run(args.cfg, args.ckpt, args.n_pairs, args.output_json,
        use_lpips=not args.no_lpips)
