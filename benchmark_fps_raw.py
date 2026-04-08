"""
FPS Benchmark — standard inference (no triplane cache).
Thin wrapper around benchmark_fps.benchmark.
"""

import argparse

from benchmark_fps import benchmark


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_frames", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--output_json", default=None)
    a = p.parse_args()
    benchmark(a.cfg, a.ckpt, a.n_frames, a.warmup, False, a.output_json)


if __name__ == "__main__":
    main()
