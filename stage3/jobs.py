#!/usr/bin/env python3
"""Stage-3 job list: debug seed first, then extra screening seeds.

Order is the scientific priority:
  No-Q / All / Late-50 / Early-50 / Middle-40 / onset / smooth
then extra seeds. Equal-budget jobs are listed separately and not launched
until M2/M4 actually differ.
"""

from __future__ import annotations

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

DEBUG_SEED = (10001,)
SCREEN_SEEDS = (10001, 20002, 30003)

# Extra seeds after the debug seed, still 3-seed screening total.
EXTRA_SEEDS = (20002, 30003)

EQUAL_BUDGET_MASKS = ("early50", "late50")


def debug_jobs():
    return [(mask, DEBUG_SEED[0], "natural") for mask in ROUND1_MASKS_PRIORITY]


def round1_jobs():
    jobs = debug_jobs()
    for seed in EXTRA_SEEDS:
        for mask in ROUND1_MASKS_PRIORITY:
            jobs.append((mask, seed, "natural"))
    return jobs


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
    args = p.parse_args()
    if args.equal_budget:
        jobs = equal_budget_jobs(SCREEN_SEEDS)
    elif args.debug_only:
        jobs = debug_jobs()
    else:
        jobs = round1_jobs()
    if args.shard >= 0:
        jobs = shard(jobs, args.shard, args.n_shards)
    for mask, seed, budget in jobs:
        print(f"{mask} {seed} {budget}")
