"""Flow-time target weights for the two-hour QAM mechanism screen."""

from __future__ import annotations

import numpy as np


SCREEN_MASKS = (
    "all",
    "early50",
    "late50",
    "noq",
    "late25",
    "remove_late25",
)


def target_weights(name: str, flow_steps: int = 10) -> np.ndarray:
    """Return unnormalized Q-edit multipliers in normalized flow time."""
    name = str(name).strip().lower().replace("-", "_")
    aliases = {
        "no_q": "noq",
        "early": "early50",
        "late": "late50",
        "last3": "late25",
        "without_last3": "remove_late25",
    }
    name = aliases.get(name, name)
    t = np.arange(int(flow_steps), dtype=np.float64) / float(flow_steps)
    if name == "all":
        w = np.ones_like(t)
    elif name == "noq":
        w = np.zeros_like(t)
    elif name == "early50":
        w = (t < 0.5).astype(np.float64)
    elif name == "late50":
        w = (t >= 0.5).astype(np.float64)
    elif name == "late25":
        w = (t >= 0.7).astype(np.float64)
    elif name == "remove_late25":
        w = (t < 0.7).astype(np.float64)
    else:
        raise ValueError(f"unknown target mask {name!r}; expected one of {SCREEN_MASKS}")
    return w


def describe_target_mask(name: str, flow_steps: int = 10) -> dict:
    name = str(name).strip().lower().replace("-", "_")
    if name == "no_q":
        name = "noq"
    w = target_weights(name, flow_steps)
    active = np.flatnonzero(w > 0).tolist()
    return {
        "name": name,
        "weights": tuple(float(x) for x in w),
        "active_steps": active,
        "n_active": len(active),
        "weight_sum": float(w.sum()),
        "summary": f"target-mask={name} T={flow_steps} active={active} w={tuple(w.tolist())}",
    }
