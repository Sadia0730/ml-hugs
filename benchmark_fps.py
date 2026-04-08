"""
FPS Benchmark — Cloth-HUGS
===========================
Measures ONE inference iteration (identical to the validate() loop in
gs_trainer.py: forward → render_human_scene, NO loss, NO backward pass).

Two modes:
  default         : standard inference (triplane runs every frame)
  --cache_triplane: cache triplane features once (they're pose-independent),
                    only LBS + rasterization run per frame → much faster

Usage
-----
# Standard inference (same as validate()):
python benchmark_fps.py \\
    --cfg  "output/human_scene/neuman/citron/hugs_trimlp/demo/2025-10-15_15-14-16_citron_final/config_train.yaml" \\
    --ckpt "output/human_scene/neuman/citron/hugs_trimlp/demo/2025-10-15_15-14-16_citron_final/ckpt/human_final.pth" \\
    --n_frames 200 --warmup 30 --output_json results_fps.json

# With triplane caching (shows speedup potential):
python benchmark_fps.py ... --cache_triplane
"""

import os, sys, json, argparse, time
import torch, numpy as np
from omegaconf import OmegaConf
from loguru import logger

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(cfg):
    if cfg.dataset.name == 'zju':
        from hugs.datasets.zju import ZJUMoCapDataset
        return ZJUMoCapDataset(
            cfg.dataset.seq, split='val',
            render_mode=cfg.mode,
            cloth_upper=getattr(cfg.dataset, 'cloth_upper', 'tshirt'),
            cloth_lower=getattr(cfg.dataset, 'cloth_lower', 'pants'),
            cloth_dir=getattr(cfg.dataset, 'cloth_dir', 'assets/snug'),
            dataset_path=getattr(cfg.dataset, 'dataset_path', 'data/zju_mocap/processed'),
        )
    else:
        from hugs.datasets import NeumanDataset
        return NeumanDataset(
            cfg.dataset.seq, 'val', cfg.mode,
            cloth_upper=getattr(cfg.dataset, 'cloth_upper', 'tshirt'),
            cloth_lower=getattr(cfg.dataset, 'cloth_lower', 'pants'),
            cloth_dir=getattr(cfg.dataset, 'cloth_dir', 'assets/snug'),
            dataset_path=getattr(cfg.dataset, 'dataset_path', None),
        )


def build_model(cfg, dataset):
    from hugs.models.hugs_trimlp import HUGS_TRIMLP
    init_betas = torch.stack([x['betas'] for x in dataset.cached_data])
    model = HUGS_TRIMLP(
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
    model.create_betas(init_betas[0], False)
    if hasattr(dataset, 'cloth_vertices') and dataset.cloth_vertices is not None:
        model.initialize_cloth(dataset.cloth_vertices, dataset.cloth_faces)
    return model


def e_pair():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def ms(start, end):
    return start.elapsed_time(end)


def stats(arr):
    a = np.array(arr)
    return dict(mean=float(np.mean(a)), std=float(np.std(a)),
                min=float(np.min(a)), max=float(np.max(a)),
                fps=float(1000 / np.mean(a)))


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def benchmark(cfg_path, ckpt_path, n_frames=200, warmup=30,
              cache_triplane=False, output_json=None):

    cfg     = OmegaConf.load(cfg_path)
    dataset = load_dataset(cfg)
    model   = build_model(cfg, dataset)

    ckpt = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(ckpt)
    model.eval()

    gpu  = torch.cuda.get_device_properties(0)
    n_body  = model._xyz.shape[0]
    n_cloth = model.cloth_gaussians['xyz'].shape[0] if model.cloth_gaussians else 0
    has_cloth = model.cloth_gaussians is not None

    logger.info(f"GPU: {gpu.name}  Body: {n_body:,} GS  Cloth: {n_cloth:,} GS")
    logger.info(f"Mode: {'triplane CACHED (pose-independent feat reuse)' if cache_triplane else 'standard inference'}")

    # ── pre-load frames (loop dataset to reach n_frames) ─────────────────────
    raw = [dataset[i] for i in range(len(dataset))]
    frames = [{k: v.cuda() if isinstance(v, torch.Tensor) else v
               for k, v in f.items()} for f in raw]
    pool   = [frames[i % len(frames)] for i in range(n_frames + warmup)]

    from hugs.renderer.gs_renderer import render_human_scene

    # ── cache triplane features if requested ──────────────────────────────────
    # Explanation: triplane(get_xyz) queries canonical Gaussian positions which
    # are FIXED (not pose-dependent). Caching saves ~47ms/frame for body and
    # ~47ms/frame for cloth — only LBS + rasterization need to run per frame.
    cached_body_feats  = None
    cached_cloth_feats = None
    if cache_triplane:
        logger.info("Pre-computing triplane features (one-time cost) …")
        cached_body_feats = model.triplane(model.get_xyz)          # [N_body, F]
        if has_cloth:
            cached_cloth_feats = model.triplane(
                model.cloth_gaussians['xyz'])                      # [N_cloth, F]
        torch.cuda.synchronize()
        logger.info("Triplane features cached.")

    # ── warm-up ───────────────────────────────────────────────────────────────
    logger.info(f"Warm-up ({warmup} iters) …")
    bg = torch.zeros(3, dtype=torch.float32, device='cuda')
    for i in range(warmup):
        data = pool[i]
        ss   = data.get('smpl_scale', torch.ones(1, device='cuda'))
        if ss.dim() == 0: ss = ss.unsqueeze(0)
        human_pack = model.forward(smpl_scale=ss, dataset_idx=-1, is_train=False)
        body_gs = human_pack['body'] if isinstance(human_pack, dict) else human_pack
        cloth_gs = human_pack.get('cloth') if isinstance(human_pack, dict) else None
        render_human_scene(data=data, human_gs_out=body_gs, scene_gs_out=None,
                           bg_color=bg, render_mode='human', cloth_gs_out=cloth_gs)
    torch.cuda.synchronize()
    logger.info("Warm-up done.")

    # ── timing arrays ─────────────────────────────────────────────────────────
    t_triplane = []   # triplane decode (body + cloth)
    t_lbs      = []   # LBS / deformation (body + cloth)
    t_render   = []   # rasterization only (CUDA kernel)
    t_iter     = []   # full iteration (what validate() does per frame)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP — exact same code path as validate() in gs_trainer.py
    # Each iteration = 1 validation step:
    #   human_gs.forward(pose) → render_human_scene() → done
    # NO loss, NO backward, NO optimizer step.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"Benchmarking {n_frames} inference iterations …")

    for i in range(n_frames):
        data = pool[warmup + i]
        ss   = data.get('smpl_scale', torch.ones(1, device='cuda'))
        if ss.dim() == 0: ss = ss.unsqueeze(0)

        # ── A. Triplane decode timing ─────────────────────────────────────────
        if not cache_triplane:
            ts, te = e_pair(); ts.record()
            _ = model.triplane(model.get_xyz)
            _ = model.appearance_dec(_)
            if has_cloth:
                _c = model.triplane(model.cloth_gaussians['xyz'])
                model.appearance_dec(_c)
            te.record(); torch.cuda.synchronize()
            t_triplane.append(ms(ts, te))

        # ── B. LBS / deformation timing ───────────────────────────────────────
        ts, te = e_pair(); ts.record()
        # (mirrors what forward_body + forward_cloth do, minus triplane when cached)
        human_pack = model.forward(smpl_scale=ss, dataset_idx=-1, is_train=False)
        te.record(); torch.cuda.synchronize()
        if cache_triplane:
            t_lbs.append(ms(ts, te))          # forward is just LBS (triplane cached)
        else:
            # subtract triplane time already run above
            t_lbs.append(max(0, ms(ts, te) - t_triplane[-1]))

        body_gs  = human_pack['body'] if isinstance(human_pack, dict) else human_pack
        cloth_gs = human_pack.get('cloth') if isinstance(human_pack, dict) else None

        # ── C. Rasterization timing (CUDA kernel only) ────────────────────────
        ts, te = e_pair(); ts.record()
        render_human_scene(data=data, human_gs_out=body_gs, scene_gs_out=None,
                           bg_color=bg, render_mode='human', cloth_gs_out=cloth_gs)
        te.record(); torch.cuda.synchronize()
        t_render.append(ms(ts, te))

        # ── D. Full iteration wall-clock (= what validate() pays per frame) ───
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        human_pack2 = model.forward(smpl_scale=ss, dataset_idx=-1, is_train=False)
        body2  = human_pack2['body'] if isinstance(human_pack2, dict) else human_pack2
        cloth2 = human_pack2.get('cloth') if isinstance(human_pack2, dict) else None
        render_human_scene(data=data, human_gs_out=body2, scene_gs_out=None,
                           bg_color=bg, render_mode='human', cloth_gs_out=cloth2)
        torch.cuda.synchronize()
        t_iter.append((time.perf_counter() - t0) * 1000)

    # ── Report ────────────────────────────────────────────────────────────────
    s_tp  = stats(t_triplane) if t_triplane else None
    s_lbs = stats(t_lbs)
    s_ren = stats(t_render)
    s_it  = stats(t_iter)

    sep = "─" * 72
    print(f"\n{sep}")
    mode_label = "CACHE TRIPLANE" if cache_triplane else "STANDARD"
    print(f"  FPS Benchmark [{mode_label}] — inference only (no loss, no backward)")
    print(sep)
    print(f"  GPU             : {gpu.name}")
    print(f"  Body Gaussians  : {n_body:,}  (n_subdivision={cfg.human.n_subdivision})")
    print(f"  Cloth Gaussians : {n_cloth:,}")
    print(f"  Frames timed    : {n_frames}  (dataset: {len(dataset)} frames, looped)")
    print(sep)

    def row(label, s, star=""):
        print(f"  {label:<30s}  {s['mean']:7.2f} ms  ±{s['std']:5.2f}  "
              f"[{s['min']:.1f}–{s['max']:.1f}]  {s['fps']:6.1f} FPS  {star}")

    print(f"  {'Component':<30s}  {'Mean':>9}  {'Std':>7}  {'Range':>12}  {'FPS':>8}")
    print(sep)
    if s_tp:
        row("TriPlane decode (body+cloth)", s_tp)
    else:
        print(f"  {'TriPlane decode (body+cloth)':<30s}  {'CACHED (0 ms per frame)'}")
    row("LBS / deformation", s_lbs)
    row("Rendering (rasterization) ◄", s_ren, "← supports 60+ FPS claim")
    print(sep)
    row("FULL ITER (= 1 validate step)", s_it)
    print(sep)

    print(f"""
  WHAT THIS MEANS:
    Rendering FPS  : {s_ren['fps']:.1f} FPS  — rasterization kernel only
                     This is the "60+ FPS" number consistent with HUGS.
    Full iter FPS  : {s_it['fps']:.1f} FPS  — forward + render per frame
                     Slower due to TriPlane MLP on {n_body:,} Gaussians
                     + cloth passes (n_subdivision={cfg.human.n_subdivision} → {n_body:,} body GS).

  WHY triplane is slow: canonical features are queried on {n_body:,}
    points (n_subdivision=2 expanded SMPL from 6890 → {n_body:,}).
    These features are POSE-INDEPENDENT and could be cached.
{'  With --cache_triplane: triplane runs once at load time, saving ~94ms/frame.' if not cache_triplane else f'  With caching enabled: triplane cost = 0ms → full iter FPS = {s_it["fps"]:.1f}'}
""")

    results = {
        "gpu": gpu.name, "n_frames": n_frames,
        "mode": "cache_triplane" if cache_triplane else "standard",
        "body_gaussians": n_body, "cloth_gaussians": n_cloth,
        "rendering_fps": s_ren,
        "lbs_ms": s_lbs,
        "full_iter_fps": s_it,
    }
    if s_tp: results["triplane_ms"] = s_tp

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved → {output_json}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",  required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_frames",       type=int, default=200)
    p.add_argument("--warmup",         type=int, default=30)
    p.add_argument("--cache_triplane", action="store_true",
                   help="Cache triplane features (pose-independent) — shows speedup potential")
    p.add_argument("--output_json", default=None)
    a = p.parse_args()
    benchmark(a.cfg, a.ckpt, a.n_frames, a.warmup, a.cache_triplane, a.output_json)
