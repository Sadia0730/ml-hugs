#
# FPS benchmark for the HUGS inference pipeline.
#
# Measures end-to-end FPS of the human + scene rendering pipeline and also
# reports a per-stage breakdown (triplane encoder, MLP decoders, SMPL/LBS
# deformation, Gaussian rasterizer) over the validation set.
#
# Usage (same CLI style as scripts/evaluate.py):
#     python scripts/benchmark_fps.py -o <path/to/output_dir>
#
# <path/to/output_dir> must contain `config_train.yaml` and the trained
# `human_final.pth` / `scene_final.pth` checkpoints (i.e. the same layout
# the pretrained models ship in).
#

import os
import sys
import glob
import json
import argparse
from statistics import mean, median

import torch
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf

sys.path.append('.')

import hugs.trainer.gs_trainer as gs_trainer_mod
from hugs.trainer import GaussianTrainer
from hugs.utils.general import safe_state
from hugs.cfg.config import cfg as default_cfg
from hugs.renderer.gs_renderer import render_human_scene


WARMUP_ITERS = 10  # GPU/kernel warmup before we start timing


class CudaTimer:
    """Context manager timing a GPU block with cudaEvents (ms)."""

    def __init__(self, bucket):
        self.bucket = bucket
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end.record()
        torch.cuda.synchronize()
        self.bucket.append(self.start.elapsed_time(self.end))


def _stats(ms_list):
    if len(ms_list) == 0:
        return {'count': 0, 'mean_ms': 0.0, 'median_ms': 0.0, 'fps': 0.0}
    m = mean(ms_list)
    return {
        'count': len(ms_list),
        'mean_ms': m,
        'median_ms': median(ms_list),
        'min_ms': min(ms_list),
        'max_ms': max(ms_list),
        'fps': 1000.0 / m if m > 0 else float('inf'),
    }


def _print_row(name, st):
    print(f"  {name:<32s} mean={st['mean_ms']:7.2f} ms  "
          f"median={st['median_ms']:7.2f} ms  "
          f"fps={st['fps']:8.2f}")


def _resolve_ckpts(cfg):
    """Mirror scripts/evaluate.py: find latest human/scene ckpts under the
    output dir. Also accepts the pretrained-models layout where the ckpts
    sit directly next to config_train.yaml (human_final.pth etc.)."""
    search_dirs = [cfg.logdir_ckpt, os.path.join(cfg.logdir_ckpt, 'ckpt'), cfg.logdir]

    def _find(pattern):
        hits = []
        for d in search_dirs:
            hits += glob.glob(os.path.join(d, pattern))
        return sorted(hits)

    human_ckpts = _find('*human*.pth')
    if human_ckpts:
        cfg.human.ckpt = human_ckpts[-1]
        logger.info(f'Found human ckpt: {cfg.human.ckpt}')
    elif cfg.mode in ['human', 'human_scene']:
        raise RuntimeError(f'Human ckpt is required for {cfg.mode} mode.')

    scene_ckpts = _find('*scene*.pth')
    if scene_ckpts:
        cfg.scene.ckpt = scene_ckpts[-1]
        logger.info(f'Found scene ckpt: {cfg.scene.ckpt}')
    elif cfg.mode in ['scene', 'human_scene']:
        raise RuntimeError(f'Scene ckpt is required for {cfg.mode} mode.')


@torch.no_grad()
def _run_frame(human_gs, scene_gs, data, bg_color, render_mode,
               t_human_fwd, t_scene_fwd, t_render, t_pipeline):
    with CudaTimer(t_pipeline):
        if human_gs is not None:
            with CudaTimer(t_human_fwd):
                human_out = human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
        else:
            human_out = None

        if scene_gs is not None:
            with CudaTimer(t_scene_fwd):
                scene_out = scene_gs.forward()
        else:
            scene_out = None

        with CudaTimer(t_render):
            _ = render_human_scene(
                data=data,
                human_gs_out=human_out,
                scene_gs_out=scene_out,
                bg_color=bg_color,
                render_mode=render_mode,
            )


@torch.no_grad()
def _time_triplane_isolated(human_gs, n, bucket):
    if human_gs is None:
        return
    xyz = human_gs.get_xyz
    for _ in range(WARMUP_ITERS):
        _ = human_gs.triplane(xyz)
    torch.cuda.synchronize()
    for _ in range(n):
        with CudaTimer(bucket):
            _ = human_gs.triplane(xyz)


@torch.no_grad()
def _time_decoders_isolated(human_gs, n, bucket):
    if human_gs is None:
        return
    tri_feats = human_gs.triplane(human_gs.get_xyz)
    for _ in range(WARMUP_ITERS):
        _ = human_gs.appearance_dec(tri_feats)
        _ = human_gs.geometry_dec(tri_feats)
        if human_gs.use_deformer:
            _ = human_gs.deformation_dec(tri_feats)
    torch.cuda.synchronize()
    for _ in range(n):
        with CudaTimer(bucket):
            _ = human_gs.appearance_dec(tri_feats)
            _ = human_gs.geometry_dec(tri_feats)
            if human_gs.use_deformer:
                _ = human_gs.deformation_dec(tri_feats)


@torch.no_grad()
def benchmark(cfg):
    trainer = GaussianTrainer(cfg)
    dataset = trainer.val_dataset
    if len(dataset) == 0:
        raise RuntimeError('Validation dataset is empty, nothing to benchmark.')

    human_gs = trainer.human_gs
    scene_gs = trainer.scene_gs
    if human_gs is not None:
        human_gs.eval()

    bg_color = torch.zeros(3, dtype=torch.float32, device='cuda')
    render_mode = cfg.mode

    t_human_fwd, t_scene_fwd, t_render, t_pipeline = [], [], [], []

    logger.info(f'Validation dataset has {len(dataset)} frames')
    logger.info(f'Running {WARMUP_ITERS} warmup iterations')
    for i in range(WARMUP_ITERS):
        data = dataset[i % len(dataset)]
        _run_frame(human_gs, scene_gs, data, bg_color, render_mode,
                   [], [], [], [])  # discard warmup timings
    torch.cuda.synchronize()

    # Walk the full validation set once (same as trainer.validate()).
    for data in tqdm(dataset, desc='FPS benchmark'):
        _run_frame(human_gs, scene_gs, data, bg_color, render_mode,
                   t_human_fwd, t_scene_fwd, t_render, t_pipeline)

    # Isolated sub-stages of the human model (micro-benchmarks, reuse the
    # same number of samples as the val walk so stats are comparable).
    t_triplane_iso, t_decoders_iso = [], []
    _time_triplane_isolated(human_gs, len(dataset), t_triplane_iso)
    _time_decoders_isolated(human_gs, len(dataset), t_decoders_iso)

    return {
        'config': {
            'num_val_frames': len(dataset),
            'warmup': WARMUP_ITERS,
            'render_mode': render_mode,
            'num_human_gaussians': int(human_gs.get_xyz.shape[0]) if human_gs is not None else 0,
            'num_scene_gaussians': int(scene_gs.get_xyz.shape[0]) if scene_gs is not None else 0,
            'image_height': int(dataset[0]['image_height']),
            'image_width': int(dataset[0]['image_width']),
        },
        'pipeline_full': _stats(t_pipeline),
        'stages': {
            'human_forward_total': _stats(t_human_fwd),
            'scene_forward': _stats(t_scene_fwd),
            'rasterization': _stats(t_render),
            'triplane_only': _stats(t_triplane_iso),
            'mlp_decoders_only': _stats(t_decoders_iso),
        },
    }


def _print_report(results):
    cfg_info = results['config']
    print()
    print('=' * 70)
    print(' HUGS inference FPS benchmark')
    print('=' * 70)
    print(f"  render_mode           : {cfg_info['render_mode']}")
    print(f"  image size            : {cfg_info['image_width']} x {cfg_info['image_height']}")
    print(f"  # human gaussians     : {cfg_info['num_human_gaussians']:,}")
    print(f"  # scene gaussians     : {cfg_info['num_scene_gaussians']:,}")
    print(f"  # val frames (timed)  : {cfg_info['num_val_frames']}  (warmup: {cfg_info['warmup']})")
    print('-' * 70)
    print(' Full pipeline (end-to-end per frame):')
    _print_row('pipeline_full', results['pipeline_full'])
    print('-' * 70)
    print(' Stage breakdown:')
    for name, st in results['stages'].items():
        if st['count'] == 0:
            continue
        _print_row(name, st)
    print('=' * 70)
    print(' Notes:')
    print('  - triplane_only / mlp_decoders_only are isolated micro-benchmarks;')
    print('    they are subsets of human_forward_total (which also includes')
    print('    SMPL forward + LBS deformation + tensor bookkeeping).')
    print('  - pipeline_full is the single-frame forward pass used to render')
    print('    one output image; multiply by frames for video throughput.')
    print()


def main():
    parser = argparse.ArgumentParser(description='FPS benchmark for HUGS inference pipeline')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Training output dir (contains config_train.yaml and the ckpts)')
    parser.add_argument('--save_json', default=None,
                        help='Optional path to dump results as JSON (default: <output_dir>/fps_benchmark.json)')
    args, extras = parser.parse_known_args()

    cfg_file = os.path.join(args.output_dir, 'config_train.yaml')
    cfg_file = OmegaConf.load(cfg_file)
    cfg = OmegaConf.merge(default_cfg, cfg_file, OmegaConf.from_cli(extras))
    cfg.eval = True
    cfg.logdir = args.output_dir
    cfg.logdir_ckpt = args.output_dir

    # The pretrained configs ship with the older human name 'hugs_triplane',
    # but gs_trainer.py only knows 'hugs_trimlp' / 'hugs_wo_trimlp'. Normalize.
    if getattr(cfg.human, 'name', None) == 'hugs_triplane':
        cfg.human.name = 'hugs_trimlp'
        logger.info("Remapped cfg.human.name: hugs_triplane -> hugs_trimlp")

    safe_state(seed=cfg.seed)
    _resolve_ckpts(cfg)

    # gs_trainer.py line ~212 does:
    #   betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas')
    #           else self.train_dataset.betas[0]
    # For the pretrained ckpts, hasattr(human_gs, 'betas') is False and the
    # NeumanDataset doesn't expose a .betas attribute either (betas live
    # per-frame inside cached_data). Plus with cfg.eval=True train_dataset
    # is not created. We patch the trainer module so __init__ can reach
    # the end without crashing:
    #   * get_train_dataset returns the (cheap) val dataset with a stub
    #     .betas attribute that satisfies line 212 only.
    #   * optimize_init is a no-op (skips the 5000-step pre-training).
    # We never call trainer.train(); val_dataset is used for the actual
    # FPS timing loop.
    cfg.eval = False

    def _stub_train_dataset(cfg_inner):
        ds = gs_trainer_mod.get_val_dataset(cfg_inner)
        # synthesize a .betas tensor from the cached per-frame betas so
        # line 212's fallback (`self.train_dataset.betas[0]`) succeeds.
        betas_list = [d['betas'] for d in ds.cached_data]
        ds.betas = torch.stack(betas_list, dim=0)
        return ds

    gs_trainer_mod.get_train_dataset = _stub_train_dataset
    gs_trainer_mod.optimize_init = lambda model, num_steps=0: model

    results = benchmark(cfg)
    _print_report(results)

    if args.save_json is None:
        args.save_json = os.path.join(args.output_dir, 'fps_benchmark.json')
    with open(args.save_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Wrote FPS benchmark results to {args.save_json}')


if __name__ == '__main__':
    main()
