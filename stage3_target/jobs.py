#!/usr/bin/env python3
"""Job list for the target-weight mechanism screen."""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stage3_target.target_weights import SCREEN_MASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    args = parser.parse_args()
    for i, mask in enumerate(SCREEN_MASKS):
        if i % args.n_shards == args.shard:
            print(mask)


if __name__ == "__main__":
    main()
