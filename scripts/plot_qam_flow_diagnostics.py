#!/usr/bin/env python3
"""Plot Stage-2 QAM flow-time raw signals (mean curve + sample distribution).

Does not synthesize a reliability score. Reads diag_dumps/*.npz and optional diag.csv.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_dumps(dump_dir):
    paths = sorted(glob.glob(os.path.join(dump_dir, "step_*.npz")))
    dumps = []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            dumps.append({k: z[k] for k in z.files})
    return dumps


def _ensure_out(out_dir):
    os.makedirs(out_dir, exist_ok=True)


def _scatter_by_t(ax, t, values, rng, alpha=0.08, s=6):
    """values: (T, N). Jitter t so the cloud is visible."""
    t = np.asarray(t).reshape(-1)
    T, n = values.shape
    tt = np.repeat(t[:, None], n, axis=1)
    jitter = (rng.rand(T, n) - 0.5) * (0.6 * (t[1] - t[0] if T > 1 else 0.04))
    ax.scatter((tt + jitter).ravel(), values.ravel(), s=s, alpha=alpha, c="C0", linewidths=0)


def _mean_quantile(ax, t, values, label, color="C0"):
    mean = values.mean(axis=1)
    q10 = np.quantile(values, 0.10, axis=1)
    q90 = np.quantile(values, 0.90, axis=1)
    ax.fill_between(t, q10, q90, color=color, alpha=0.22, label="q10–q90")
    ax.plot(t, mean, color=color, lw=2.2, label=label)
    ax.plot(t, np.quantile(values, 0.50, axis=1), color=color, lw=1.2, ls="--", alpha=0.8, label="median")


def _panel_t_vs(ax, t, values, title, ylabel, rng):
    _scatter_by_t(ax, t, values, rng)
    _mean_quantile(ax, t, values, "mean")
    ax.set_title(title)
    ax.set_xlabel("flow time t")
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(t.min()) - 0.05, float(t.max()) + 0.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _stack_over_steps(dumps, key):
    """Return steps (S,), t (T,), values (S, T, N)."""
    steps = np.array([int(d["step"]) for d in dumps])
    t = dumps[0]["t"].astype(np.float64)
    stacked = np.stack([np.asarray(d[key], dtype=np.float64) for d in dumps], axis=0)
    return steps, t, stacked


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    print("wrote", path)


def plot_abc(dumps, out_dir, rng):
    latest = dumps[-1]
    t = latest["t"].astype(np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    _panel_t_vs(axes[0], t, np.asarray(latest["p_norm"], dtype=np.float64), "A  t vs ||p_t||", r"$\|p_t\|$", rng)
    _panel_t_vs(axes[1], t, np.asarray(latest["R_grad_t"], dtype=np.float64), "B  t vs $R_{grad}$", r"$R_{\mathrm{grad}}$", rng)
    _panel_t_vs(axes[2], t, np.asarray(latest["vf_ratio"], dtype=np.float64), r"C  t vs $\|f_\theta\|/(\|f_\beta\|+\epsilon)$", r"$\|f_\theta\|/(\|f_\beta\|+\epsilon)$", rng)
    fig.suptitle(f"mean + sample distribution  (latest dump step={int(latest['step'])})", y=1.03)
    _save(fig, os.path.join(out_dir, "figABC_t_vs_p_Rgrad_vfratio.png"))

    # variance callout
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))
    for ax, key, name in zip(
        axes,
        ("p_norm", "R_grad_t", "vf_ratio"),
        (r"Var(\|p_t\| | t)", r"Var(R_grad | t)", r"Var(vf_ratio | t)"),
    ):
        v = np.asarray(latest[key], dtype=np.float64)
        ax.plot(t, v.var(axis=1), lw=2)
        ax.set_title(name)
        ax.set_xlabel("flow time t")
        ax.grid(True, alpha=0.25)
    _save(fig, os.path.join(out_dir, "figABC_conditional_variance.png"))


def plot_flow_extras(latest, out_dir, rng):
    t = latest["t"].astype(np.float64)
    panels = [
        ("x_norm", r"$\|x_t\|$", "figD_x_norm"),
        ("v_fast_norm", r"$\|v^{\mathrm{fast}}_t\|$", "figE_v_fast"),
        ("v_slow_norm", r"$\|v^{\mathrm{slow}}_t\|$", "figE2_v_slow"),
        ("ell", r"$\ell_t$", "figF_ell"),
        ("vf_rel_edit", r"$\|f_\theta-f_\beta\|/(\|f_\beta\|+\epsilon)$", "figC2_rel_edit"),
        ("q_xt_mean", r"$\bar Q(s, x_t)$", "figG_Qbar_xt"),
        ("q_xt_std", r"$\mathrm{Std}_i Q_i(s,x_t)$", "figG2_Qstd_xt"),
        ("U_grad_t", r"$U_{\mathrm{grad}}(t)$", "figB2_Ugrad"),
        ("cos_p_pterm", r"$\cos(p_t, p_{T-})$", "figH_cos_p_pterm"),
        ("cos_p_gmean", r"$\cos(p_t, \bar g_t)$", "figH2_cos_p_g"),
        ("x_support_min", r"min L2$(x_t$, minibatch $a)$", "figI_x_support"),
        ("g_xt_norm_mean", r"mean$_i \|\nabla_x Q_i\|$", "figJ_g_norm"),
    ]
    for key, ylab, fname in panels:
        if key not in latest:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        _panel_t_vs(ax, t, np.asarray(latest[key], dtype=np.float64), fname.split("_", 1)[-1], ylab, rng)
        _save(fig, os.path.join(out_dir, fname + ".png"))


def plot_policy_and_term(dumps, out_dir, rng):
    latest = dumps[-1]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.3))
    for ax, key, name in zip(
        axes,
        ("policy_drift", "A_rl_support_min", "U_grad_term"),
        (r"$\|A_{\mathrm{RL}}-A_{\mathrm{base}}\|$", r"min L2$(A_{\mathrm{RL}}$, minibatch)", r"$U_{\mathrm{grad}}$ at $A$"),
    ):
        v = np.asarray(latest[key], dtype=np.float64).reshape(-1)
        ax.hist(v, bins=24, color="C0", alpha=0.75)
        ax.set_title(f"{name}\nmean={v.mean():.3g}  std={v.std():.3g}")
        ax.set_xlabel(name)
        ax.grid(True, alpha=0.25)
    _save(fig, os.path.join(out_dir, "figK_terminal_policy_hist.png"))

    if len(dumps) < 2:
        return
    steps = np.array([int(d["step"]) for d in dumps])

    def series(key):
        return np.array([np.mean(np.asarray(d[key], dtype=np.float64)) for d in dumps])

    def band(key):
        q10 = np.array([np.quantile(np.asarray(d[key], dtype=np.float64), 0.1) for d in dumps])
        q90 = np.array([np.quantile(np.asarray(d[key], dtype=np.float64), 0.9) for d in dumps])
        return q10, q90

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    specs = [
        (axes[0, 0], "policy_drift", r"$\|A_{\mathrm{RL}}-A_{\mathrm{base}}\|$"),
        (axes[0, 1], "A_rl_support_min", r"support proxy of $A_{\mathrm{RL}}$"),
        (axes[1, 0], "U_grad_term", r"$U_{\mathrm{grad}}(A)$"),
        (axes[1, 1], "R_grad_term", r"$R_{\mathrm{grad}}(A)$"),
    ]
    for ax, key, ylab in specs:
        m = series(key)
        q10, q90 = band(key)
        ax.fill_between(steps, q10, q90, alpha=0.22)
        ax.plot(steps, m, lw=2)
        ax.set_ylabel(ylab)
        ax.set_xlabel("train step")
        ax.grid(True, alpha=0.25)
    _save(fig, os.path.join(out_dir, "figL_policy_drift_vs_step.png"))


def plot_heatmaps(dumps, out_dir):
    if len(dumps) < 2:
        return
    keys = [
        ("p_norm", r"mean $\|p_t\|$"),
        ("R_grad_t", r"mean $R_{\mathrm{grad}}(t)$"),
        ("U_grad_t", r"mean $U_{\mathrm{grad}}(t)$"),
        ("vf_ratio", r"mean $\|f_\theta\|/(\|f_\beta\|+\epsilon)$"),
        ("ell", r"mean $\ell_t$"),
        ("q_xt_mean", r"mean $\bar Q(s,x_t)$"),
        ("q_xt_std", r"mean Std$(Q_i)$"),
        ("x_support_min", r"mean support dist of $x_t$"),
        ("cos_p_pterm", r"mean $\cos(p_t,p_{T-})$"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.5))
    for ax, (key, title) in zip(axes.ravel(), keys):
        steps, t, stacked = _stack_over_steps(dumps, key)
        mean_st = stacked.mean(axis=-1)
        im = ax.imshow(
            mean_st,
            aspect="auto",
            origin="lower",
            extent=[float(t.min()), float(t.max()), float(steps.min()), float(steps.max())],
            cmap="magma",
        )
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("step")
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, os.path.join(out_dir, "figM_step_by_t_heatmaps.png"))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    for ax, key, title in zip(
        axes,
        ("p_norm", "R_grad_t", "vf_ratio"),
        (r"std$_\mathrm{batch}\|p_t\|$", r"std$_\mathrm{batch} R_{\mathrm{grad}}$", r"std$_\mathrm{batch}$ vf_ratio"),
    ):
        steps, t, stacked = _stack_over_steps(dumps, key)
        im = ax.imshow(
            stacked.std(axis=-1),
            aspect="auto",
            origin="lower",
            extent=[float(t.min()), float(t.max()), float(steps.min()), float(steps.max())],
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("step")
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, os.path.join(out_dir, "figM2_step_by_t_batch_std.png"))


def plot_stage_overlay(dumps, out_dir):
    """Early / mid / late mean±q-band for A/B/C — when Q becomes useful."""
    if len(dumps) == 1:
        picks = dumps
        labels = [f"step {int(dumps[0]['step'])}"]
    else:
        idxs = [0, len(dumps) // 2, len(dumps) - 1]
        picks = [dumps[i] for i in idxs]
        labels = [f"step {int(d['step'])}" for d in picks]
    t = dumps[0]["t"].astype(np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    colors = ["C0", "C1", "C2"]
    keys = [("p_norm", r"$\|p_t\|$"), ("R_grad_t", r"$R_{\mathrm{grad}}$"), ("vf_ratio", r"$\|f_\theta\|/(\|f_\beta\|+\epsilon)$")]
    for ax, (key, ylab) in zip(axes, keys):
        for d, lab, c in zip(picks, labels, colors):
            v = np.asarray(d[key], dtype=np.float64)
            ax.fill_between(t, np.quantile(v, 0.1, axis=1), np.quantile(v, 0.9, axis=1), color=c, alpha=0.12)
            ax.plot(t, v.mean(axis=1), color=c, lw=2, label=lab)
        ax.set_xlabel("t")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("early / mid / late  (mean + q10–q90)")
    _save(fig, os.path.join(out_dir, "figN_early_mid_late_ABC.png"))


def plot_q_ensemble(latest, out_dir):
    q = np.asarray(latest["q_term"], dtype=np.float64)
    g = np.asarray(latest["g_term"], dtype=np.float64)
    m, n = q.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    axes[0].boxplot([q[i] for i in range(m)], showfliers=False)
    axes[0].set_xlabel("critic i")
    axes[0].set_ylabel(r"$Q_i(s,A)$")
    axes[0].set_title("terminal ensemble Q")
    gnorm = np.linalg.norm(g, axis=-1)
    axes[1].boxplot([gnorm[i] for i in range(m)], showfliers=False)
    axes[1].set_xlabel("critic i")
    axes[1].set_ylabel(r"$\|\nabla_A Q_i\|$")
    axes[1].set_title("terminal ensemble |grad|")
    _save(fig, os.path.join(out_dir, "figO_terminal_ensemble.png"))


def plot_scatter_evidence(latest, out_dir):
    t = latest["t"].astype(np.float64)
    T = t.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    cmap = plt.cm.viridis
    for k in range(T):
        ax.scatter(
            np.asarray(latest["p_norm"][k], dtype=np.float64),
            np.asarray(latest["U_grad_t"][k], dtype=np.float64),
            s=8,
            alpha=0.25,
            color=cmap(k / max(T - 1, 1)),
            label=f"t={t[k]:.1f}" if k % 2 == 0 else None,
        )
    ax.set_xlabel(r"$\|p_t\|$")
    ax.set_ylabel(r"$U_{\mathrm{grad}}(t)$")
    ax.set_title("adjoint magnitude vs gradient disagreement")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    _save(fig, os.path.join(out_dir, "figP_pnorm_vs_Ugrad.png"))


def write_summary_table(dumps, out_dir):
    latest = dumps[-1]
    t = latest["t"].astype(np.float64)
    lines = ["t,p_norm_mean,p_norm_std,R_grad_mean,R_grad_std,U_grad_mean,vf_ratio_mean,vf_ratio_std,ell_mean,Qbar_mean,Qstd_mean,var_R_grad"]
    for k, tk in enumerate(t):
        pn = np.asarray(latest["p_norm"][k], dtype=np.float64)
        rg = np.asarray(latest["R_grad_t"][k], dtype=np.float64)
        ug = np.asarray(latest["U_grad_t"][k], dtype=np.float64)
        vr = np.asarray(latest["vf_ratio"][k], dtype=np.float64)
        ell = np.asarray(latest["ell"][k], dtype=np.float64)
        qb = np.asarray(latest["q_xt_mean"][k], dtype=np.float64)
        qs = np.asarray(latest["q_xt_std"][k], dtype=np.float64)
        lines.append(
            f"{tk:.3f},{pn.mean():.6g},{pn.std():.6g},{rg.mean():.6g},{rg.std():.6g},{ug.mean():.6g},"
            f"{vr.mean():.6g},{vr.std():.6g},{ell.mean():.6g},{qb.mean():.6g},{qs.mean():.6g},{rg.var():.6g}"
        )
    path = os.path.join(out_dir, "latest_t_moments.csv")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Agent save_dir containing diag_dumps/")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    dump_dir = os.path.join(args.run_dir, "diag_dumps")
    out_dir = args.out_dir or os.path.join(args.run_dir, "plots")
    _ensure_out(out_dir)
    dumps = _load_dumps(dump_dir)
    if not dumps:
        raise SystemExit(f"no npz dumps in {dump_dir}")
    rng = np.random.RandomState(0)
    plot_abc(dumps, out_dir, rng)
    plot_flow_extras(dumps[-1], out_dir, rng)
    plot_policy_and_term(dumps, out_dir, rng)
    plot_heatmaps(dumps, out_dir)
    plot_stage_overlay(dumps, out_dir)
    plot_q_ensemble(dumps[-1], out_dir)
    plot_scatter_evidence(dumps[-1], out_dir)
    write_summary_table(dumps, out_dir)
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(
            "Raw-signal plots only. No reliability score.\n"
            "A: adjoint magnitude vs flow time.\n"
            "B: critic-gradient direction agreement vs flow time.\n"
            "C: how much QAM edits the base field vs flow time.\n"
            "If E[C_t] trends with t but Var(C_t|t) is huge, time is related to "
            "improvement structure but is not a sufficient statistic for using Q.\n"
        )
    print(f"done: {len(dumps)} dumps -> {out_dir}")


if __name__ == "__main__":
    main()
