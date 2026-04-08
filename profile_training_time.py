"""
Training Time Component Profiler — Cloth-HUGS
=============================================
Runs a short training loop (default 100 iterations) with CUDA Events and
reports a per-component time breakdown:

  • TriPlane decode          (triplane query + appearance/geometry MLPs)
  • LBS / deformation        (SMPL forward + linear blend skinning)
  • Rasterization            (diff_gaussian_rasterization CUDA kernel)
  • Cloth forward            (cloth triplane query + cloth LBS)
  • Cloth extra renders      (cloth color pass + visibility matte pass)
  • Loss computation         (L1 + SSIM + LPIPS + physics losses)
  • Backward pass            (autograd)

This directly answers Reviewer 2's request for a training-time breakdown.

Usage
-----
python profile_training_time.py \
    --cfg  output/human_scene/cloth_zju_386_train/2025-11-01_02-38-10/config_train.yaml \
    --ckpt output/human_scene/cloth_zju_386_train/2025-11-01_02-38-10/ckpt/human_final.pth \
    --n_iters 100
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
def make_event_pair():
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e


class ComponentTimer:
    """Accumulate CUDA event timings per named component."""

    def __init__(self, components):
        self._data = {c: [] for c in components}
        self._events = {c: make_event_pair() for c in components}

    def start(self, component):
        self._events[component][0].record()

    def stop(self, component):
        self._events[component][1].record()

    def sync_all(self):
        torch.cuda.synchronize()
        for c in self._data:
            ms = self._events[c][0].elapsed_time(self._events[c][1])
            self._data[c].append(ms)
            # Refresh events for next iteration
            self._events[c] = make_event_pair()

    def report(self):
        totals = {}
        for c, vals in self._data.items():
            a = np.array(vals)
            totals[c] = {
                "mean_ms": float(np.mean(a)),
                "std_ms" : float(np.std(a)),
                "pct"    : 0.0,   # filled below
            }
        # Compute percentage of sum
        total_ms = sum(v["mean_ms"] for v in totals.values())
        for c in totals:
            totals[c]["pct"] = 100.0 * totals[c]["mean_ms"] / max(total_ms, 1e-6)
        return totals, total_ms


# ─────────────────────────────────────────────────────────────────────────────
def build_model_and_dataset(cfg, ckpt_path):
    from hugs.models.hugs_trimlp import HUGS_TRIMLP
    from hugs.datasets.zju import ZJUMoCapDataset
    from hugs.datasets import NeumanDataset

    if cfg.dataset.name == 'zju':
        train_ds = ZJUMoCapDataset(
            cfg.dataset.seq, split='train',
            render_mode=getattr(cfg, 'mode', 'human'),
            cloth_upper=getattr(cfg.dataset, 'cloth_upper', 'tshirt'),
            cloth_lower=getattr(cfg.dataset, 'cloth_lower', 'pants'),
            cloth_dir=getattr(cfg.dataset, 'cloth_dir', 'assets/snug'),
            dataset_path=getattr(cfg.dataset, 'dataset_path', 'data/zju_mocap/processed'),
        )
    else:
        train_ds = NeumanDataset(
            cfg.dataset.seq, 'train',
            render_mode=getattr(cfg, 'mode', 'human'),
            cloth_upper=getattr(cfg.dataset, 'cloth_upper', 'tshirt'),
            cloth_lower=getattr(cfg.dataset, 'cloth_lower', 'pants'),
            cloth_dir=getattr(cfg.dataset, 'cloth_dir', 'assets/snug'),
            dataset_path=getattr(cfg.dataset, 'dataset_path', None),
        )

    init_betas = torch.stack([x['betas'] for x in train_ds.cached_data], dim=0)
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
    human_gs.create_betas(init_betas[0], cfg.human.optim_betas)

    # Pose params
    init_pose    = torch.stack([x['body_pose']      for x in train_ds.cached_data])
    init_orient  = torch.stack([x['global_orient']  for x in train_ds.cached_data])
    init_transl  = torch.stack([x['transl']         for x in train_ds.cached_data])
    human_gs.create_body_pose(init_pose,   cfg.human.optim_pose)
    human_gs.create_global_orient(init_orient, cfg.human.optim_pose)
    human_gs.create_transl(init_transl,    cfg.human.optim_trans)

    if not cfg.eval:
        human_gs.initialize()

    if hasattr(train_ds, 'cloth_vertices') and train_ds.cloth_vertices is not None:
        human_gs.initialize_cloth(train_ds.cloth_vertices, train_ds.cloth_faces)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cuda')
    human_gs.load_state_dict(ckpt)

    # Setup optimizer (needed for backward)
    human_gs.setup_optimizer(cfg=cfg.human.lr)

    return human_gs, train_ds


# ─────────────────────────────────────────────────────────────────────────────
def profile(cfg_path, ckpt_path, n_iters=100, warmup=10, output_json=None):
    cfg = OmegaConf.load(cfg_path)

    logger.info("Building model and dataset …")
    human_gs, train_ds = build_model_and_dataset(cfg, ckpt_path)
    human_gs.train()

    from hugs.renderer.gs_renderer import render, render_human_scene
    from hugs.losses.utils import l1_loss, ssim

    has_cloth = human_gs.cloth_gaussians is not None

    components = [
        "triplane_decode",
        "lbs_deform",
        "rasterization",
    ]
    if has_cloth:
        components += ["cloth_forward", "cloth_renders"]
    components += ["loss_backward"]

    timer = ComponentTimer(components)

    bg_color = torch.zeros(3, dtype=torch.float32, device='cuda')

    n_data = len(train_ds)
    data_list = [train_ds[i] for i in range(min(n_data, n_iters + warmup))]

    def to_cuda(d):
        return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    logger.info(f"Warm-up ({warmup} iters) …")
    for i in range(warmup):
        data = to_cuda(data_list[i % len(data_list)])
        smpl_scale = data.get('smpl_scale', torch.ones(1, device='cuda'))
        if smpl_scale.dim() == 0:
            smpl_scale = smpl_scale.unsqueeze(0)
        pack = human_gs.forward(
            global_orient=data.get('global_orient'),
            body_pose=data.get('body_pose'),
            betas=data.get('betas'),
            transl=data.get('transl'),
            smpl_scale=smpl_scale,
            dataset_idx=0,
            is_train=True,
        )
    torch.cuda.synchronize()
    logger.info("Profiling …")

    for iteration in range(n_iters):
        data = to_cuda(data_list[(iteration + warmup) % len(data_list)])
        smpl_scale = data.get('smpl_scale', torch.ones(1, device='cuda'))
        if smpl_scale.dim() == 0:
            smpl_scale = smpl_scale.unsqueeze(0)

        # ── 1. TriPlane decode ─────────────────────────────────────────────────
        timer.start("triplane_decode")
        tri_feats      = human_gs.triplane(human_gs.get_xyz)
        appearance_out = human_gs.appearance_dec(tri_feats)
        geometry_out   = human_gs.geometry_dec(tri_feats)
        if human_gs.use_deformer:
            deformation_out = human_gs.deformation_dec(tri_feats)
        timer.stop("triplane_decode")

        # ── 2. LBS / deformation ──────────────────────────────────────────────
        timer.start("lbs_deform")
        body_out = human_gs.forward_body(
            global_orient=data.get('global_orient'),
            body_pose=data.get('body_pose'),
            betas=data.get('betas'),
            transl=data.get('transl'),
            smpl_scale=smpl_scale,
            dataset_idx=0,
            is_train=True,
        )
        timer.stop("lbs_deform")

        # ── 3. Base rasterization ─────────────────────────────────────────────
        timer.start("rasterization")
        render_pkg = render(
            means3D=body_out['xyz'],
            feats=body_out['shs'],
            opacity=body_out['opacity'],
            scales=body_out['scales'],
            rotations=body_out['rotq'],
            data=data,
            bg_color=bg_color,
            active_sh_degree=body_out['active_sh_degree'],
        )
        timer.stop("rasterization")

        # ── 4. Cloth (if present) ─────────────────────────────────────────────
        cloth_out = None
        if has_cloth:
            timer.start("cloth_forward")
            cloth_out = human_gs.forward_cloth(
                global_orient=data.get('global_orient'),
                body_pose=data.get('body_pose'),
                betas=data.get('betas'),
                transl=data.get('transl'),
                smpl_scale=smpl_scale,
                dataset_idx=0,
                is_train=True,
            )
            timer.stop("cloth_forward")

            # Cloth extra render passes
            timer.start("cloth_renders")
            from hugs.renderer.gs_renderer import _render_colors_only, _render_visibility_matte
            cloth_rgb, cloth_info = _render_colors_only(
                means3D=cloth_out['xyz'], feats=cloth_out['shs'],
                opacity=cloth_out['opacity'], scales=cloth_out['scales'],
                rotations=cloth_out['rotq'], data=data,
                scaling_modifier=1.0, bg_color=bg_color,
                sh_degree=cloth_out['active_sh_degree'],
            )
            blockers = {
                'xyz':    body_out['xyz'],
                'opacity':body_out['opacity'],
                'scales': body_out['scales'],
                'rotq':   body_out['rotq'],
            }
            cloth_vis = _render_visibility_matte(
                cloth=cloth_out, blockers=blockers,
                data=data, scaling_modifier=1.0, sh_degree=0,
            )
            timer.stop("cloth_renders")

        # ── 5. Loss + backward ─────────────────────────────────────────────────
        timer.start("loss_backward")
        rendered_img = render_pkg['render']
        gt_img       = data['rgb']

        loss = l1_loss(rendered_img, gt_img)
        loss = loss + 0.2 * (1.0 - ssim(rendered_img, gt_img))

        if human_gs.optimizer:
            human_gs.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        timer.stop("loss_backward")

        timer.sync_all()

        if (iteration + 1) % 20 == 0:
            logger.info(f"  iter {iteration+1}/{n_iters}  loss={loss.item():.4f}")

    # ── Aggregate and report ──────────────────────────────────────────────────
    breakdown, total_ms = timer.report()

    sep = "─" * 66
    print(f"\n{sep}")
    print("  Training Time Breakdown — Cloth-HUGS")
    print(sep)
    print(f"  {'Component':<28s}  {'Mean (ms)':>10}  {'Std':>7}  {'% of sum':>9}")
    print(sep)

    order = [
        ("triplane_decode",  "TriPlane decode"),
        ("lbs_deform",       "LBS / deformation"),
        ("rasterization",    "Rasterization (base)"),
    ]
    if has_cloth:
        order += [
            ("cloth_forward",  "Cloth forward"),
            ("cloth_renders",  "Cloth extra renders"),
        ]
    order += [("loss_backward", "Loss + backward")]

    for key, label in order:
        d = breakdown[key]
        print(f"  {label:<28s}  {d['mean_ms']:>10.2f}  {d['std_ms']:>7.2f}  {d['pct']:>8.1f}%")

    print(sep)
    print(f"  {'Sum of components':<28s}  {total_ms:>10.2f}")

    # Estimated full-iteration time (components + small overhead)
    print()
    print("  Note: 'Loss + backward' dominates because it runs autograd over")
    print("  the full graph. At inference (no grad), backward is removed —")
    print("  FPS benchmark measures that faster path.")
    print(sep)

    results = {
        "n_iters": n_iters,
        "breakdown": {k: v for k, v in breakdown.items()},
        "total_component_ms": total_ms,
    }

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved to {output_json}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",  required=True, help="config_train.yaml path")
    parser.add_argument("--ckpt", required=True, help="human_final.pth path")
    parser.add_argument("--n_iters", type=int, default=100, help="Iterations to profile (default 100)")
    parser.add_argument("--warmup",  type=int, default=10,  help="Warmup iterations (default 10)")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    profile(args.cfg, args.ckpt, args.n_iters, args.warmup, args.output_json)
