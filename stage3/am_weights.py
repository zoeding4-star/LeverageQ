"""Flow-time adjoint-matching weights for Stage 3.

Time convention (matches QAM's integrator):

    t in [0, 1],   t_k = k / T,   k = 0, ..., T-1

t=0 is source / initial noise; t -> 1 is the terminal action. With T=10
evaluation points this is {0, 0.1, ..., 0.9}. Never refer to "step 3" in
reports; always use normalized t and the active-step count.

Natural (hard) masks keep w_k in {0, 1} and do **not** re-normalize, so the
number of AM terms can be smaller than T. Equal-budget masks scale by
T / sum(w) so sum_k w'_k = T.

Smooth schedules (early-heavy / late-heavy / quad-late-heavy) are always
scaled to mean(w)=1, i.e. the same total Q-guidance budget as All.

Stage 3B floor+boost (never zero except No-Q):

    floor = 0.25,   peak_1 = 1.0,   peak_+0.25 = 0.50
    increasing: affine floor -> 1,  decreasing: affine 1 -> floor
"""

from __future__ import annotations

import numpy as np

# T=10 discrete masks from the Stage-3 protocol (source of truth).
_T10 = {
    "noq": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
    "all": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64),
    "early50": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float64),
    "middle40": np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.float64),
    "late50": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64),
    # 8 active steps (~75%); not an exact continuous 75%.
    "late75": np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64),
    # 3 active steps (~25%); not an exact continuous 25%.
    "late25": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.float64),
}

MASK_IDS = {
    "noq": "M0",
    "all": "M1",
    "early50": "M2",
    "middle40": "M3",
    "late50": "M4",
    "late75": "M5",
    "late25": "M6",
    "early_heavy": "M7",
    "late_heavy": "M8",
    "quad_late_heavy": "M9",
    "early50_norm": "M2n",
    "late50_norm": "M4n",
}

ALIASES = {
    "no_q": "noq",
    "m0": "noq",
    "uniform": "all",
    "m1": "all",
    "early_50": "early50",
    "early": "early50",
    "m2": "early50",
    "middle_40": "middle40",
    "middle": "middle40",
    "m3": "middle40",
    "late_50": "late50",
    "late": "late50",
    "m4": "late50",
    "late_75": "late75",
    "m5": "late75",
    "late_25": "late25",
    "m6": "late25",
    "earlyheavy": "early_heavy",
    "m7": "early_heavy",
    "lateheavy": "late_heavy",
    "m8": "late_heavy",
    "quad_late": "quad_late_heavy",
    "m9": "quad_late_heavy",
    "const_half": "const05",
    "const_0.5": "const05",
    "constant05": "const05",
    "half": "const05",
    "early_boost1": "early_b1",
    "middle_boost1": "middle_b1",
    "late_boost1": "late_b1",
    "early_boost025": "early_b025",
    "middle_boost025": "middle_b025",
    "late_boost025": "late_b025",
    "early_plus025": "early_b025",
    "inc": "increasing",
    "dec": "decreasing",
}

# Stage 3B: weak non-zero floor + region boost. Not equal-budget.
STAGE3B_FLOOR = 0.25
STAGE3B_PEAK_1 = 1.0
STAGE3B_PEAK_025 = STAGE3B_FLOOR + 0.25  # additive +0.25 boost -> 0.50
STAGE3B_IDS = {
    "noq": "B0",
    "all": "B1",
    "const05": "B2",
    "early_b1": "B3",
    "middle_b1": "B4",
    "late_b1": "B5",
    "early_b025": "B6",
    "middle_b025": "B7",
    "late_b025": "B8",
    "increasing": "B9",
    "decreasing": "B10",
}
STAGE3B_ROUND = (
    "noq",
    "all",
    "const05",
    "late_b1",
    "early_b1",
    "middle_b1",
    "late_b025",
    "early_b025",
    "middle_b025",
    "increasing",
    "decreasing",
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n", ""):
        return False
    raise ValueError(f"cannot parse bool from {value!r}")


def normalize_mask_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    return ALIASES.get(key, key)


def normalized_flow_times(flow_steps: int) -> np.ndarray:
    """t_k = k / T for k = 0..T-1."""
    t = int(flow_steps)
    if t <= 0:
        raise ValueError(f"flow_steps must be positive, got {flow_steps}")
    return np.arange(t, dtype=np.float64) / float(t)


def _hard_from_predicate(t: np.ndarray, pred) -> np.ndarray:
    return pred(t).astype(np.float64)


def _region_bool(t: np.ndarray, region: str) -> np.ndarray:
    if region == "early":
        return t < 0.5
    if region == "middle":
        return (t >= 0.3) & (t < 0.7)
    if region == "late":
        return t >= 0.5
    raise ValueError(f"unknown region {region!r}")


def floor_boost_weights(
    region: str,
    flow_steps: int = 10,
    floor: float = STAGE3B_FLOOR,
    peak: float = STAGE3B_PEAK_1,
) -> np.ndarray:
    """w = floor everywhere, peak on the named region. min(w)=floor > 0."""
    t = normalized_flow_times(flow_steps)
    floor = float(floor)
    peak = float(peak)
    if floor <= 0.0:
        raise ValueError(f"floor must be > 0, got {floor}")
    w = np.full(t.shape, floor, dtype=np.float64)
    w[_region_bool(t, region)] = peak
    return w


def ramp_weights(
    flow_steps: int = 10,
    lo: float = STAGE3B_FLOOR,
    hi: float = STAGE3B_PEAK_1,
    decreasing: bool = False,
) -> np.ndarray:
    """Affine in t_k from lo to hi (or hi to lo). Endpoints stay > 0."""
    t = normalized_flow_times(flow_steps)
    lo = float(lo)
    hi = float(hi)
    if min(lo, hi) <= 0.0:
        raise ValueError(f"ramp endpoints must be > 0, got lo={lo} hi={hi}")
    if t.size == 1:
        return np.array([hi if decreasing else lo], dtype=np.float64)
    u = t / t[-1]
    if decreasing:
        return hi + (lo - hi) * u
    return lo + (hi - lo) * u


def compute_am_weights(
    name: str,
    flow_steps: int = 10,
    eps: float = 1e-3,
    budget_normalize: bool = False,
) -> np.ndarray:
    """Return w with shape (T,).

    Hard masks: 0/1. Smooth schedules: mean(w)=1. If budget_normalize and
    sum(w)>0, rescale so sum(w)=T (All-step budget). No-Q stays all zeros.
    """
    name = normalize_mask_name(name)
    T = int(flow_steps)
    t = normalized_flow_times(T)
    eps = float(eps)
    budget_normalize = _as_bool(budget_normalize)

    if name in _T10 and T == 10 and name not in ("early_heavy", "late_heavy", "quad_late_heavy"):
        w = _T10[name].copy()
    elif name == "noq":
        w = np.zeros(T, dtype=np.float64)
    elif name in ("all", "uniform"):
        w = np.ones(T, dtype=np.float64)
    elif name == "early50":
        w = _hard_from_predicate(t, lambda x: x < 0.5)
    elif name == "middle40":
        w = _hard_from_predicate(t, lambda x: (x >= 0.3) & (x < 0.7))
    elif name == "late50":
        w = _hard_from_predicate(t, lambda x: x >= 0.5)
    elif name == "late75":
        w = _hard_from_predicate(t, lambda x: x >= 0.2)
    elif name == "late25":
        w = _hard_from_predicate(t, lambda x: x >= 0.7)
    elif name == "early_heavy":
        w = (1.0 - t + eps)
        w = w / w.mean()
    elif name == "late_heavy":
        w = (t + eps)
        w = w / w.mean()
    elif name == "quad_late_heavy":
        w = (t + eps) ** 2
        w = w / w.mean()
    elif name in ("const05", "const_half"):
        w = np.full(T, 0.5, dtype=np.float64)
    elif name == "early_b1":
        w = floor_boost_weights("early", T, STAGE3B_FLOOR, STAGE3B_PEAK_1)
    elif name == "middle_b1":
        w = floor_boost_weights("middle", T, STAGE3B_FLOOR, STAGE3B_PEAK_1)
    elif name == "late_b1":
        w = floor_boost_weights("late", T, STAGE3B_FLOOR, STAGE3B_PEAK_1)
    elif name == "early_b025":
        w = floor_boost_weights("early", T, STAGE3B_FLOOR, STAGE3B_PEAK_025)
    elif name == "middle_b025":
        w = floor_boost_weights("middle", T, STAGE3B_FLOOR, STAGE3B_PEAK_025)
    elif name == "late_b025":
        w = floor_boost_weights("late", T, STAGE3B_FLOOR, STAGE3B_PEAK_025)
    elif name == "increasing":
        w = ramp_weights(T, STAGE3B_FLOOR, STAGE3B_PEAK_1, decreasing=False)
    elif name == "decreasing":
        w = ramp_weights(T, STAGE3B_FLOOR, STAGE3B_PEAK_1, decreasing=True)
    else:
        known = sorted(set(list(_T10) + list(MASK_IDS) + list(STAGE3B_IDS)))
        raise ValueError(f"Unknown am_mask_name={name!r}. Known: {known}")

    if budget_normalize and float(w.sum()) > 0.0:
        w = w * (T / float(w.sum()))
    return w


def region_masks(flow_steps: int = 10):
    """Boolean masks for F_signal regions, keyed by name."""
    t = normalized_flow_times(flow_steps)
    return {
        "early": t < 0.5,
        "middle": (t >= 0.3) & (t < 0.7),
        "late": t >= 0.5,
    }


def f_signal(p_norm_t: np.ndarray, region: np.ndarray, eps: float = 1e-8) -> float:
    """F_signal(S) = sum_{t in S} ||p_t|| / sum_t ||p_t||.

    p_norm_t: (T,) batch-mean adjoint norms.
    """
    p = np.asarray(p_norm_t, dtype=np.float64).reshape(-1)
    m = np.asarray(region, dtype=np.float64).reshape(-1)
    denom = float(p.sum()) + eps
    return float((p * m).sum() / denom)


def describe_am_mask(
    name: str,
    flow_steps: int = 10,
    eps: float = 1e-3,
    budget_normalize: bool = False,
) -> dict:
    w = compute_am_weights(name, flow_steps=flow_steps, eps=eps, budget_normalize=budget_normalize)
    t = normalized_flow_times(flow_steps)
    name = normalize_mask_name(name)
    budget_normalize = _as_bool(budget_normalize)
    active = w > 0
    n_active = int(active.sum())
    if n_active > 0:
        t_lo = float(t[active].min())
        t_hi = float(t[active].max())
        active_steps = np.nonzero(active)[0].tolist()
    else:
        t_lo, t_hi, active_steps = 0.0, 0.0, []
    mid = MASK_IDS.get(name, STAGE3B_IDS.get(name, name))
    if budget_normalize and name in ("early50", "late50"):
        mid = MASK_IDS.get(f"{name}_norm", mid)
    weights = tuple(float(x) for x in w)
    summary = (
        f"{mid} {name} T={flow_steps} n_active={n_active} "
        f"t_active=[{t_lo:.2f},{t_hi:.2f}] sum(w)={float(w.sum()):.4f} "
        f"budget_norm={budget_normalize} w={weights}"
    )
    return {
        "name": name,
        "id": mid,
        "flow_steps": int(flow_steps),
        "t": tuple(float(x) for x in t),
        "weights": weights,
        "n_active": n_active,
        "active_steps": active_steps,
        "t_lo": t_lo,
        "t_hi": t_hi,
        "weight_sum": float(w.sum()),
        "budget_normalize": bool(budget_normalize),
        "summary": summary,
    }


# First-round natural-mask matrix (no equal-budget).
STAGE3_ROUND1 = (
    "noq",
    "all",
    "early50",
    "middle40",
    "late50",
    "late75",
    "late25",
    "early_heavy",
    "late_heavy",
)
