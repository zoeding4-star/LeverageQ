"""Reduce / save QAM diagnostic snapshots. Training code does not depend on plots."""

from __future__ import annotations

import os

import numpy as np


VECTOR_KEYS = ("x", "p", "g_xt", "g_term", "A_rl", "A_base", "x_term", "dataset_actions")
PER_T_SAMPLE_KEYS = (
    "x_norm",
    "p_norm",
    "v_fast_norm",
    "v_slow_norm",
    "v_int_norm",
    "ell",
    "vf_ratio",
    "vf_rel_edit",
    "q_xt_mean",
    "q_xt_std",
    "R_grad_t",
    "U_grad_t",
    "g_xt_norm_mean",
    "cos_p_pterm",
    "cos_p_gmean",
    "x_support_min",
    "x_support_knn5",
)
PER_SAMPLE_KEYS = (
    "q_term_mean",
    "q_term_std",
    "R_grad_term",
    "U_grad_term",
    "policy_drift",
    "A_rl_support_min",
    "A_rl_support_knn5",
    "A_base_support_min",
    "A_base_support_knn5",
)


def _to_numpy(tree):
    out = {}
    for k, v in tree.items():
        out[k] = np.asarray(v)
    return out


def _moment_dict(prefix, arr):
    arr = np.asarray(arr, dtype=np.float64)
    flat = arr.reshape(-1)
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std()),
        f"{prefix}_q10": float(np.quantile(flat, 0.10)),
        f"{prefix}_q50": float(np.quantile(flat, 0.50)),
        f"{prefix}_q90": float(np.quantile(flat, 0.90)),
        f"{prefix}_var": float(flat.var()),
    }


def reduce_snapshot(snap):
    """Flatten one snapshot into JSON/CSV/W&B scalars. No reliability score."""
    snap = _to_numpy(snap)
    t = snap["t"]
    T = int(t.shape[0])
    row = {"flow_steps": float(T)}
    for k in range(T):
        pfx = f"t{k:02d}"
        row[f"{pfx}/t"] = float(t[k])
        for name in PER_T_SAMPLE_KEYS:
            row.update(_moment_dict(f"{pfx}/{name}", snap[name][k]))
        q = snap["q_xt"][k]
        row.update(_moment_dict(f"{pfx}/Q_bar", q.mean(axis=0)))
        row.update(_moment_dict(f"{pfx}/Q_std_ens", q.std(axis=0)))

    for name in PER_SAMPLE_KEYS:
        row.update(_moment_dict(name, snap[name]))
    row.update(_moment_dict("Q_bar_term", snap["q_term_mean"]))
    row.update(_moment_dict("Q_std_ens_term", snap["q_term_std"]))
    for i in range(snap["q_term"].shape[0]):
        row[f"Q_i{i:02d}_term_mean"] = float(np.mean(snap["q_term"][i]))
        row[f"g_i{i:02d}_term_norm_mean"] = float(np.mean(np.linalg.norm(snap["g_term"][i], axis=-1)))
    return row


# Axis of the training-batch dimension in diagnostic_snapshot outputs.
_BATCH_AXIS = {
    "x": 1,
    "p": 1,
    "x_term": 0,
    "x_norm": 1,
    "p_norm": 1,
    "v_fast_norm": 1,
    "v_slow_norm": 1,
    "v_int_norm": 1,
    "ell": 1,
    "vf_ratio": 1,
    "vf_rel_edit": 1,
    "q_xt": 2,
    "g_xt": 2,
    "q_xt_mean": 1,
    "q_xt_std": 1,
    "R_grad_t": 1,
    "U_grad_t": 1,
    "g_xt_norm_mean": 1,
    "q_term": 1,
    "g_term": 1,
    "q_term_mean": 0,
    "q_term_std": 0,
    "R_grad_term": 0,
    "U_grad_term": 0,
    "g_term_norm": 1,
    "cos_p_pterm": 1,
    "cos_p_gmean": 1,
    "A_rl": 0,
    "A_base": 0,
    "policy_drift": 0,
    "A_rl_support_min": 0,
    "A_rl_support_knn5": 0,
    "A_base_support_min": 0,
    "A_base_support_knn5": 0,
    "x_support_min": 1,
    "x_support_knn5": 1,
    "dataset_actions": 0,
}


def subsample_snapshot(snap, n_samples, seed):
    snap = _to_numpy(snap)
    b = int(snap["x"].shape[1])
    n = min(int(n_samples), b)
    rng = np.random.RandomState(int(seed) % (2**31 - 1))
    idx = np.sort(rng.choice(b, size=n, replace=False))
    out = {"t": snap["t"].astype(np.float32), "sample_index": idx.astype(np.int32)}
    for k, v in snap.items():
        if k == "t":
            continue
        arr = np.asarray(v)
        axis = _BATCH_AXIS.get(k)
        if axis is None:
            out[k] = arr
            continue
        out[k] = np.take(arr, idx, axis=axis)
    for k in VECTOR_KEYS:
        if k in out:
            out[k] = np.asarray(out[k], dtype=np.float16)
    return out


def save_snapshot_npz(path, snap, step, n_samples, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dumped = subsample_snapshot(snap, n_samples=n_samples, seed=seed)
    dumped["step"] = np.int64(step)
    np.savez_compressed(path, **dumped)
    return path
