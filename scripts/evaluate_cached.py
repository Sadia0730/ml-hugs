#
# Evaluate with precompute_lbs_cache() active — verifies that the LBS cache
# optimization does not change image quality (PSNR, SSIM, LPIPS, FID).
#

import os
import sys
import json
import glob
import torch
import argparse
from loguru import logger
from omegaconf import OmegaConf

sys.path.append('.')

from hugs.trainer import GaussianTrainer
from hugs.utils.general import safe_state
from hugs.cfg.config import cfg as default_cfg
from benchmark_fps import install_canonical_cache, benchmark


@torch.no_grad()
def main(cfg):
    safe_state(seed=cfg.seed)

    logger.add(os.path.join(cfg.logdir, 'eval_cached.log'), level='INFO')
    logger.info('Evaluating WITH precompute_lbs_cache() active')

    human_ckpt_files = sorted(
        glob.glob(cfg.logdir_ckpt + '/*human*.pth') +
        glob.glob(cfg.logdir_ckpt + '/ckpt/*human*.pth')
    )
    if human_ckpt_files:
        cfg.human.ckpt = human_ckpt_files[-1]
        logger.info(f'Found human ckpt: {cfg.human.ckpt}')
    else:
        logger.error('No human ckpt found'); exit()

    scene_ckpt_files = sorted(
        glob.glob(cfg.logdir_ckpt + '/*scene*.pth') +
        glob.glob(cfg.logdir_ckpt + '/ckpt/*scene*.pth')
    )
    if scene_ckpt_files:
        cfg.scene.ckpt = scene_ckpt_files[-1]
        logger.info(f'Found scene ckpt: {cfg.scene.ckpt}')
    else:
        if cfg.mode in ['scene', 'human_scene']:
            logger.error('No scene ckpt found'); exit()

    trainer = GaussianTrainer(cfg)

    # ── Activate the same caches used by benchmark_fps_cached.py ─────────────
    model = trainer.human_gs
    has_cloth = model.cloth_gaussians is not None

    logger.info('Installing canonical triplane+decoder cache …')
    install_canonical_cache(model, has_cloth)

    logger.info('Precomputing LBS cache …')
    model.precompute_lbs_cache()
    logger.info('Caches active — running validation …')
    # ─────────────────────────────────────────────────────────────────────────

    trainer.validate()
    # animate() uses dataset_idx-based pose lookup which is incompatible with
    # the LBS cache — skip it, validate() is sufficient for quality verification

    # ── Also measure FPS with same caches active ──────────────────────────────
    logger.info('Measuring FPS with caches active …')
    fps_results = benchmark(cfg._cfg_path, cfg.human.ckpt, n_frames=200, warmup=30,
                            cache_triplane=True, output_json=None)

    results = {
        'image_metrics': trainer.eval_metrics,
        'fps': {
            'full_iter_fps': fps_results['full_iter_fps'],
            'rendering_fps': fps_results['rendering_fps'],
            'lbs_ms': fps_results['lbs_ms'],
            'gpu': fps_results['gpu'],
        }
    }

    out_path = os.path.join(cfg.logdir, 'results_eval_cached.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f'Done. Results saved to {out_path}')
    logger.info(f'Image metrics: {trainer.eval_metrics}')
    logger.info(f"FPS: {fps_results['full_iter_fps']['fps']:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="path to config yaml")
    parser.add_argument("--ckpt", required=True, help="path to human_final.pth")
    parser.add_argument("--output_dir", default=None, help="where to save results (default: next to ckpt)")
    args, extras = parser.parse_known_args()

    cfg_file = OmegaConf.load(args.cfg)
    cfg = OmegaConf.merge(default_cfg, cfg_file, OmegaConf.from_cli(extras))
    cfg.eval = True
    cfg.human.ckpt = args.ckpt
    cfg._cfg_path = args.cfg  # store for FPS benchmark

    out_dir = args.output_dir or os.path.dirname(os.path.dirname(args.ckpt))
    cfg.logdir = out_dir
    cfg.logdir_ckpt = os.path.dirname(args.ckpt)

    main(cfg)
