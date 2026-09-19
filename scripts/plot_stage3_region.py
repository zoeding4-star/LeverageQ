#!/usr/bin/env python3
"""Stage-3 plots: mask weights, Stage-2 F_signal prior, and success curves.

Does not need W&B. Reads local eval.csv / offline_agent.csv when present.
"""

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
    STAGE3_ROUND1,
    compute_am_weights,
    describe_am_mask,
    f_signal,
    normalized_flow_times,
    region_masks,
)

DEFAULT_OUT = os.path.join(ROOT, "notes/figures/stage3_cube_double")
STAGE2_MOMENTS = os.path.join(ROOT, "notes/figures/stage2_cube_double/latest_t_moments.csv")

MASK_LABELS = {
    "noq": "M0 No-Q",
    "all": "M1 All",
    "early50": "M2 Early-50",
    "middle40": "M3 Middle-40",
    "late50": "M4 Late-50",
    "late75": "M5 Late-75",
    "late25": "M6 Late-25",
    "early_heavy": "M7 Early-heavy",
    "late_heavy": "M8 Late-heavy",
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
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.5), sharex=True, sharey=False)
    for ax, name in zip(axes.ravel(), STAGE3_ROUND1):
        w = compute_am_weights(name)
        spec = describe_am_mask(name)
        ax.step(t, w, where="mid", lw=2.2, color="C0")
        ax.scatter(t, w, s=28, color="C0", zorder=3)
        ax.set_title(f"{spec['id']}  {name}\nactive={spec['n_active']}  sum(w)={spec['weight_sum']:.2f}")
        ax.set_xlim(-0.05, 0.95)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("flow time t = k/T")
        ax.set_ylabel(r"$w_k$")
    fig.suptitle("Stage 3 natural / standard scaling  (T=10, t_k=k/10)", y=1.01)
    _save(fig, os.path.join(out_dir, "fig_masks_natural.png"))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for name in ("all", "early50", "late50"):
        ax.step(t, compute_am_weights(name, budget_normalize=True), where="mid", lw=2, label=f"{name} equal-budget")
    ax.set_title("Equal-budget control (not launched until M2/M4 differ)")
    ax.set_xlabel("flow time t")
    ax.set_ylabel(r"$w'_k = T w_k / \sum w$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, os.path.join(out_dir, "fig_masks_equal_budget.png"))


def plot_stage2_f_signal_prior(out_dir):
    if not os.path.exists(STAGE2_MOMENTS):
        print("skip F_signal prior; missing", STAGE2_MOMENTS)
        return
    rows = list(csv.DictReader(open(STAGE2_MOMENTS)))
    t = np.array([float(r["t"]) for r in rows])
    p = np.array([float(r["p_norm_mean"]) for r in rows])
    m = region_masks(10)
    fe, fm, fl = f_signal(p, m["early"]), f_signal(p, m["middle"]), f_signal(p, m["late"])
    f_l75 = f_signal(p, compute_am_weights("late75") > 0)
    f_l25 = f_signal(p, compute_am_weights("late25") > 0)
    f_e50 = f_signal(p, compute_am_weights("early50") > 0)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(t, p, lw=2.4, marker="o", label=r"Stage-2 $\mathbb{E}\|p_t\|$")
    ax.axvspan(-0.05, 0.45, color="C0", alpha=0.08, label=f"early F={fe:.2f}")
    ax.axvspan(0.45, 0.95, color="C3", alpha=0.08, label=f"late F={fl:.2f}")
    ax.set_xlabel("flow time t")
    ax.set_ylabel(r"$\mathbb{E}\|p_t\|$")
    ax.set_title("Stage-2 prior: adjoint magnitude is late-concentrated")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, os.path.join(out_dir, "fig_stage2_Fsignal_prior.png"))

    path = os.path.join(out_dir, "stage2_Fsignal_prior.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "active_steps", "F_signal"])
        w.writerow(["early50", "0-4", f"{f_e50:.6f}"])
        w.writerow(["middle40", "3-6", f"{fm:.6f}"])
        w.writerow(["late50", "5-9", f"{fl:.6f}"])
        w.writerow(["late75", "2-9", f"{f_l75:.6f}"])
        w.writerow(["late25", "7-9", f"{f_l25:.6f}"])
    print("wrote", path)


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
        if "stage3-region" not in str(group):
            continue
        if "smoke" in str(group):
            continue
        rows = list(csv.DictReader(open(eval_path)))
        if not rows or "success" not in rows[0]:
            continue
        steps = np.array([float(r["step"]) for r in rows])
        succ = np.array([float(r["success"]) for r in rows])
        runs.append({"dir": run_dir, "mask": mask, "seed": seed, "group": group, "step": steps, "success": succ})
    return runs


def plot_success(runs, out_dir):
    if not runs:
        print("no full Stage-3 eval.csv yet; skip success plots")
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    colors = {m: f"C{i}" for i, m in enumerate(STAGE3_ROUND1)}
    for mask in STAGE3_ROUND1:
        subset = [r for r in runs if r["mask"] == mask]
        if not subset:
            continue
        for r in subset:
            ax.plot(r["step"], r["success"], color=colors[mask], alpha=0.28, lw=1.0)
        # mean over seeds at shared eval grid of the longest
        grid = subset[0]["step"]
        mats = []
        for r in subset:
            if len(r["step"]) == len(grid) and np.allclose(r["step"], grid):
                mats.append(r["success"])
        if mats:
            mean = np.mean(np.stack(mats, 0), 0)
            ax.plot(grid, mean, color=colors[mask], lw=2.3, label=MASK_LABELS.get(mask, mask))
        else:
            ax.plot(subset[0]["step"], subset[0]["success"], color=colors[mask], lw=2.3, label=MASK_LABELS.get(mask, mask))
    ax.set_xlabel("train step")
    ax.set_ylabel("success")
    ax.set_title("Stage 3 cube-double-task2  Success(k)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, os.path.join(out_dir, "fig_success_curves.png"))

    # onset curve at 1M and 1.5M
    onset = [("all", 0.0), ("late75", 0.2), ("late50", 0.5), ("late25", 0.7)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for target, marker, label in ((1_000_000, "o", "offline 1M"), (1_500_000, "s", "online 1.5M")):
        xs, ys = [], []
        for mask, tau in onset:
            vals = []
            for r in runs:
                if r["mask"] != mask:
                    continue
                if target in r["step"]:
                    vals.append(float(r["success"][np.where(r["step"] == target)[0][0]]))
                elif len(r["step"]):
                    idx = int(np.argmin(np.abs(r["step"] - target)))
                    if abs(r["step"][idx] - target) <= 1000:
                        vals.append(float(r["success"][idx]))
            if vals:
                xs.append(tau)
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, marker=marker, lw=2, label=label)
    ax.set_xlabel(r"guidance onset $\tau_s$  ($w(t)=1[t\ge\tau_s]$)")
    ax.set_ylabel("success")
    ax.set_title("Success vs guidance start time")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, os.path.join(out_dir, "fig_onset_curve.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument(
        "--exp-root",
        default=os.path.join(ROOT, "exp/qam-reproduce"),
    )
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_weights(args.out)
    plot_stage2_f_signal_prior(args.out)
    runs = _load_eval_runs(args.exp_root)
    plot_success(runs, args.out)


if __name__ == "__main__":
    main()
