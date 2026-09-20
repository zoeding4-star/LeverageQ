#!/usr/bin/env python3
"""Stage-3 job lists.

round1: original hard-mask screening (cube-double-task2).
stage3b: floor+boost on cube-double-task4 (never-zero except No-Q).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stage3.am_weights import STAGE3B_ROUND

DEBUG_SEED = (10001,)
SCREEN_SEEDS = (10001, 20002, 30003)
EXTRA_SEEDS = (20002, 30003)

EQUAL_BUDGET_MASKS = ("early50", "late50")

# Keep the old priority order used by existing tmux workers.
ROUND1_MASKS_PRIORITY = (
    "noq",
    "all",
    "late50",
    "early50",
    "middle40",
    "late75",
    "late25",
    "early_heavy",
    "late_heavy",
)


def debug_jobs():
    return [(mask, DEBUG_SEED[0], "natural") for mask in ROUND1_MASKS_PRIORITY]


def round1_jobs():
    jobs = debug_jobs()
    for seed in EXTRA_SEEDS:
        for mask in ROUND1_MASKS_PRIORITY:
            jobs.append((mask, seed, "natural"))
    return jobs


def stage3b_jobs(seeds=DEBUG_SEED):
    return [(mask, int(seed), "natural") for seed in seeds for mask in STAGE3B_ROUND]


def equal_budget_jobs(seeds=DEBUG_SEED):
    jobs = []
    for seed in seeds:
        for mask in EQUAL_BUDGET_MASKS:
            jobs.append((mask, seed, "norm"))
    return jobs


def shard(jobs, index, n_shards):
    return [j for i, j in enumerate(jobs) if i % n_shards == index]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=-1)
    p.add_argument("--n-shards", type=int, default=3)
    p.add_argument("--equal-budget", action="store_true")
    p.add_argument("--debug-only", action="store_true")
    p.add_argument("--round", choices=("round1", "stage3b"), default="round1")
    args = p.parse_args()
    if args.equal_budget:
        jobs = equal_budget_jobs(SCREEN_SEEDS)
    elif args.round == "stage3b":
        seeds = SCREEN_SEEDS if (not args.debug_only) else DEBUG_SEED
        jobs = stage3b_jobs(seeds)
    elif args.debug_only:
        jobs = debug_jobs()
    else:
        jobs = round1_jobs()
    if args.shard >= 0:
        jobs = shard(jobs, args.shard, args.n_shards)
    for mask, seed, budget in jobs:
        print(f"{mask} {seed} {budget}")
