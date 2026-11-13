#!/usr/bin/env python3
"""Synthetic CPU timing for preprocessing + ONNX + decode; excludes ROS and cloud projection."""
import argparse
import time

import numpy as np

from dr_spaam_inference import OnnxDetector, decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--beams', type=int, default=490)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--scans', type=int, default=30)
    args = parser.parse_args()
    if min(args.beams, args.threads, args.stride, args.scans) < 1 or args.beams < 2:
        parser.error('Require positive counts and at least two beams')
    detector = OnnxDetector(args.model, args.threads, args.stride)
    rng = np.random.default_rng(3)
    scan = rng.uniform(0.3, 15., args.beams).astype(np.float32)
    angles = -1.919862 + np.arange(0, args.beams, args.stride) * 0.00785
    valid = np.ones(len(angles), dtype=bool)
    timings = []
    for step in range(args.scans + 5):
        start = time.perf_counter()
        logits, offsets = detector.infer(scan, 0.00785)
        decode(scan[::args.stride], angles, valid, logits, offsets)
        if step >= 5:
            timings.append((time.perf_counter() - start) * 1000)
    print(f'CPU threads={args.threads}, beams={args.beams}, stride={args.stride}, '
          f'n={args.scans}: median={np.median(timings):.2f} ms, '
          f'p95={np.percentile(timings, 95):.2f} ms')


if __name__ == '__main__':
    main()
