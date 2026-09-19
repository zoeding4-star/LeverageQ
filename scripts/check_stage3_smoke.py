#!/usr/bin/env python3
"""Verify Stage-3 smoke CSVs: weights, finite losses, No-Q adj_loss ~ 0."""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

ROOT = "/mnt/zoe/projects/qam"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stage3.am_weights import STAGE3_ROUND1, compute_am_weights, describe_am_mask
SMOKE_GLOB = os.path.join(
    ROOT,
    "exp/qam-reproduce/stage3-region-smoke/**/offline_agent.csv",
)


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _finite(rows, keys):
    for row in rows:
        for k in keys:
            if k not in row:
                continue
            try:
                v = float(row[k])
            except (TypeError, ValueError):
                return False, f"non-float {k}={row[k]!r}"
            if not math.isfinite(v):
                return False, f"non-finite {k}={v}"
    return True, ""


def main():
    paths = sorted(glob.glob(SMOKE_GLOB, recursive=True))
    if not paths:
        print("No smoke CSVs found at", SMOKE_GLOB)
        return 2

    print(f"found {len(paths)} smoke agent CSVs")
    errors = []
    seen = set()
    check_keys = [
        "actor/am_weight_sum",
        "actor/adj_loss",
        "actor/flow_loss",
        "critic/q_mean",
        "grad/norm",
    ]
    for path in paths:
        run_dir = os.path.dirname(path)
        flags = os.path.join(run_dir, "flags.json")
        mask = None
        if os.path.exists(flags):
            with open(flags) as f:
                blob = json.load(f)
            agent = blob.get("agent") or {}
            mask = agent.get("am_mask_name")
        rows = _read_csv(path)
        if not rows:
            errors.append(f"empty csv {path}")
            continue
        ok, msg = _finite(rows, check_keys)
        if not ok:
            errors.append(f"{msg} in {path}")
            continue
        last = rows[-1]
        if "actor/am_weight_sum" not in last or mask is None:
            errors.append(f"missing mask/weight_sum in {path}")
            continue
        want = float(compute_am_weights(mask).sum())
        got = float(last["actor/am_weight_sum"])
        if abs(got - want) > 1e-3:
            errors.append(f"{mask} weight_sum {got} != {want} ({path})")
        seen.add(mask)
        if mask == "noq":
            adj = float(last["actor/adj_loss"])
            if adj > 1e-5:
                errors.append(f"noq adj_loss={adj} should be ~0 ({path})")
        # Hard masks: check a couple of w_k columns if present.
        w = compute_am_weights(mask)
        for k, wk in enumerate(w):
            col = f"actor/t{k:02d}/w"
            if col in last and abs(float(last[col]) - float(wk)) > 1e-4:
                errors.append(f"{mask} {col}={last[col]} != {wk}")
        print(
            f"ok {str(mask):12s} steps={last.get('step')} "
            f"flow={float(last['actor/flow_loss']):.4f} "
            f"adj={float(last['actor/adj_loss']):.4f} "
            f"sum_w={got:.3f}"
        )

    missing = [m for m in STAGE3_ROUND1 if m not in seen]
    if missing:
        errors.append(f"missing smoke masks: {missing}")
    if errors:
        print("SMOKE CHECK FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("SMOKE CHECK PASSED", len(seen), "masks")
    for m in STAGE3_ROUND1:
        print(" ", describe_am_mask(m)["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
