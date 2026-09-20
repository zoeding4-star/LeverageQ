#!/usr/bin/env python3
"""Stage-3B plots: floor+boost weights and success curves on cube-double-task4."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

ROOT = "/mnt/zoe/projects/qam"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage3.am_weights import (
    STAGE3B_ROUND,
    compute_am_weights,
    describe_am_mask,
    normalized_flow_times,
)

DEFAULT_OUT = os.path.join(ROOT, "notes/figures/stage3b_cube_double_task4")

MASK_LABELS = {
    "noq": "B0 No-Q",
    "all": "B1 All (w=1)",
    "const05": "B2 Const 0.5",
    "late_b1": "B5 Late peak=1",
    "early_b1": "B3 Early peak=1",
    "middle_b1": "B4 Mid peak=1",
    "late_b025": "B8 Late +0.25",
    "early_b025": "B6 Early +0.25",
    "middle_b025": "B7 Mid +0.25",
    "increasing": "B9 Increasing",
    "decreasing": "B10 Decreasing",
}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    print("wrote", path)


def plot_weights(out_dir):
    t = normalized_flow_times(10)
    fig, axes = plt.subplots(3, 4, figsize=(14.5, 9.0), sharex=True)
    axes = axes.ravel()
    for ax, name in zip(axes, STAGE3B_ROUND):
        w = compute_am_weights(name)
        spec = describe_am_mask(name)
        ax.step(t, w, where="mid", lw=2.2, color="C0")
        ax.scatter(t, w, s=28, color="C0", zorder=3)
        ax.axhline(0.25, color="0.6", ls="--", lw=0.8)
        ax.axhline(0.5, color="0.7", ls=":", lw=0.8)
        ax.set_title(f"{spec['id']}  {name}\nmin={w.min():.2f}  sum={spec['weight_sum']:.2f}")
        ax.set_xlim(-0.05, 0.95)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("flow time t = k/T")
        ax.set_ylabel(r"$w_k$")
    for ax in axes[len(STAGE3B_ROUND) :]:
        ax.axis("off")
    fig.suptitle("Stage 3B floor+boost  (floor=0.25, never zero except No-Q)", y=1.02)
    _save(fig, os.path.join(out_dir, "fig_masks_floor_boost.png"))


def _load_eval_runs(exp_root):
    runs = []
    for flags_path in glob.glob(os.path.join(exp_root, "**", "flags.json"), recursive=True):
        run_dir = os.path.dirname(flags_path)
        eval_path = os.path.join(run_dir, "eval.csv")
        if not os.path.exists(eval_path):
            continue
        with open(flags_path) as f:
            flags = json.load(f)
        agent = flags.get("agent") or {}
        mask = agent.get("am_mask_name")
        seed = flags.get("seed")
        group = flags.get("run_group", "")
        env = flags.get("env_name", "")
        if str(group) not in ("full", "stage3b-floor-boost"):
            continue
        if "smoke" in str(group):
            continue
        rows = list(csv.DictReader(open(eval_path)))
        if not rows or "success" not in rows[0]:
            continue
        steps = np.array([float(r["step"]) for r in rows])
        succ = np.array([float(r["success"]) for r in rows])
        runs.append(
            {
                "dir": run_dir,
                "mask": mask,
                "seed": seed,
                "group": group,
                "env": env,
                "step": steps,
                "success": succ,
            }
        )
    return runs


def plot_success(runs, out_dir):
    if not runs:
        print("no full Stage-3B eval.csv yet; skip success plots")
        return
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    colors = {m: f"C{i}" for i, m in enumerate(STAGE3B_ROUND)}
    for mask in STAGE3B_ROUND:
        subset = [r for r in runs if r["mask"] == mask]
        if not subset:
            continue
        for r in subset:
            ax.plot(r["step"], r["success"], color=colors[mask], alpha=0.28, lw=1.0)
        grid = subset[0]["step"]
        mats = []
        for r in subset:
            if len(r["step"]) == len(grid) and np.allclose(r["step"], grid):
                mats.append(r["success"])
        if mats:
            mean = np.mean(np.stack(mats, 0), 0)
            ax.plot(grid, mean, color=colors[mask], lw=2.3, label=MASK_LABELS.get(mask, mask))
        else:
            ax.plot(
                subset[0]["step"],
                subset[0]["success"],
                color=colors[mask],
                lw=2.3,
                label=MASK_LABELS.get(mask, mask),
            )
    ax.axvline(1_000_000, color="0.5", ls="--", lw=1.0, label="offline | online")
    ax.set_xlabel("train step")
    ax.set_ylabel("success")
    ax.set_title("Stage 3B cube-double-task4  Success(k)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, os.path.join(out_dir, "fig_success_curves.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--exp-root", default=os.path.join(ROOT, "exp/qam-stage3b"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_weights(args.out)
    runs = _load_eval_runs(args.exp_root)
    plot_success(runs, args.out)


if __name__ == "__main__":
    main()
