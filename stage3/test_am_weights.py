#!/usr/bin/env python3
"""CPU tests for Stage-3 AM weight schedules. No GPU / QAM training."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stage3.am_weights import (
    STAGE3B_FLOOR,
    STAGE3B_PEAK_025,
    STAGE3B_PEAK_1,
    STAGE3B_ROUND,
    compute_am_weights,
    describe_am_mask,
    f_signal,
    floor_boost_weights,
    normalized_flow_times,
    region_masks,
)


def _assert_close(a, b, tol=1e-9, msg=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.allclose(a, b, atol=tol, rtol=tol):
        raise AssertionError(f"{msg} {a} != {b}")


def test_bool_parse():
    from stage3.am_weights import _as_bool

    assert _as_bool(False) is False
    assert _as_bool("False") is False
    assert _as_bool("false") is False
    assert _as_bool(True) is True
    assert _as_bool("True") is True
    w = compute_am_weights("late50", budget_normalize="False")
    assert abs(w.sum() - 5.0) < 1e-12


def test_t_grid():
    t = normalized_flow_times(10)
    _assert_close(t, np.arange(10) / 10.0, msg="t_k")
    assert t[0] == 0.0


def test_t10_hard_masks():
    _assert_close(compute_am_weights("noq"), np.zeros(10))
    _assert_close(compute_am_weights("all"), np.ones(10))
    _assert_close(compute_am_weights("early50"), [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    _assert_close(compute_am_weights("middle40"), [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    _assert_close(compute_am_weights("late50"), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    _assert_close(compute_am_weights("late75"), [0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _assert_close(compute_am_weights("late25"), [0, 0, 0, 0, 0, 0, 0, 1, 1, 1])


def test_natural_no_renorm():
    w = compute_am_weights("early50", budget_normalize=False)
    assert abs(w.sum() - 5.0) < 1e-12
    w2 = compute_am_weights("late50", budget_normalize=False)
    assert abs(w2.sum() - 5.0) < 1e-12
    w3 = compute_am_weights("middle40", budget_normalize=False)
    assert abs(w3.sum() - 4.0) < 1e-12


def test_equal_budget():
    w = compute_am_weights("late50", budget_normalize=True)
    _assert_close(w, [0, 0, 0, 0, 0, 2, 2, 2, 2, 2])
    assert abs(w.sum() - 10.0) < 1e-12
    w = compute_am_weights("early50", budget_normalize=True)
    _assert_close(w, [2, 2, 2, 2, 2, 0, 0, 0, 0, 0])
    z = compute_am_weights("noq", budget_normalize=True)
    _assert_close(z, np.zeros(10))


def test_smooth_mean_one():
    for name in ("early_heavy", "late_heavy", "quad_late_heavy", "all"):
        w = compute_am_weights(name)
        assert abs(w.mean() - 1.0) < 1e-12, name
        assert abs(w.sum() - 10.0) < 1e-12, name
        assert np.all(w > 0)


def test_late_heavy_increases():
    w = compute_am_weights("late_heavy")
    assert np.all(np.diff(w) > 0)
    we = compute_am_weights("early_heavy")
    assert np.all(np.diff(we) < 0)


def test_aliases():
    _assert_close(compute_am_weights("M0"), compute_am_weights("noq"))
    _assert_close(compute_am_weights("uniform"), compute_am_weights("all"))


def test_f_signal_stage2_prior():
    # End-of-training means from notes/figures/stage2_cube_double/latest_t_moments.csv
    p = np.array(
        [0.01017, 0.05024, 0.15048, 0.33077, 0.68738, 1.14031, 1.81763, 3.22509, 5.64173, 7.77412]
    )
    m = region_masks(10)
    fe = f_signal(p, m["early"])
    fm = f_signal(p, m["middle"])
    fl = f_signal(p, m["late"])
    assert 0.04 < fe < 0.08, fe
    assert 0.15 < fm < 0.25, fm
    assert 0.90 < fl < 0.96, fl
    w_late = compute_am_weights("late50")
    f_active = f_signal(p, w_late > 0)
    _assert_close(f_active, fl, tol=1e-6)


def test_describe():
    d = describe_am_mask("late25")
    assert d["n_active"] == 3
    assert d["id"] == "M6"
    assert d["active_steps"] == [7, 8, 9]


def test_stage3b_never_zero_except_noq():
    for name in STAGE3B_ROUND:
        w = compute_am_weights(name)
        if name == "noq":
            _assert_close(w, np.zeros(10), msg=name)
        else:
            assert np.all(w > 0), name
            assert float(w.min()) >= STAGE3B_FLOOR - 1e-12, (name, w.min())


def test_const05():
    _assert_close(compute_am_weights("const05"), np.full(10, 0.5))
    _assert_close(compute_am_weights("half"), np.full(10, 0.5))


def test_floor_boost_b1():
    t = normalized_flow_times(10)
    w = compute_am_weights("early_b1")
    _assert_close(w[t < 0.5], STAGE3B_PEAK_1)
    _assert_close(w[t >= 0.5], STAGE3B_FLOOR)
    w = compute_am_weights("late_b1")
    _assert_close(w[t >= 0.5], STAGE3B_PEAK_1)
    _assert_close(w[t < 0.5], STAGE3B_FLOOR)
    w = compute_am_weights("middle_b1")
    mid = (t >= 0.3) & (t < 0.7)
    _assert_close(w[mid], STAGE3B_PEAK_1)
    _assert_close(w[~mid], STAGE3B_FLOOR)


def test_floor_boost_plus025():
    t = normalized_flow_times(10)
    w = compute_am_weights("late_b025")
    _assert_close(w[t >= 0.5], STAGE3B_PEAK_025)
    _assert_close(w[t < 0.5], STAGE3B_FLOOR)
    assert abs(STAGE3B_PEAK_025 - 0.5) < 1e-12


def test_ramps():
    inc = compute_am_weights("increasing")
    dec = compute_am_weights("decreasing")
    assert np.all(np.diff(inc) > 0)
    assert np.all(np.diff(dec) < 0)
    assert abs(inc[0] - STAGE3B_FLOOR) < 1e-12
    assert abs(inc[-1] - STAGE3B_PEAK_1) < 1e-12
    _assert_close(inc[::-1], dec)
    assert np.all(inc > 0) and np.all(dec > 0)


def test_floor_helper_rejects_zero_floor():
    try:
        floor_boost_weights("late", floor=0.0, peak=1.0)
    except ValueError:
        return
    raise AssertionError("zero floor should be rejected")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"passed {len(tests)} tests")
    for name in ("noq", "all", "early50", "middle40", "late50", "late75", "late25", "early_heavy", "late_heavy"):
        print(describe_am_mask(name)["summary"])
    print("late50_norm", describe_am_mask("late50", budget_normalize=True)["summary"])
    print("---- stage3b ----")
    for name in STAGE3B_ROUND:
        print(describe_am_mask(name)["summary"])
