"""Bound a one-hidden-layer ReLU network over a box with MILP + LP enumeration."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


def network(W, b, v, bias, lower, upper):
    W, b, v, lower, upper = (np.asarray(z, dtype=float) for z in (W, b, v, lower, upper))
    if (
        W.ndim != 2
        or min(W.shape) < 1
        or b.shape != (W.shape[0],)
        or v.shape != b.shape
        or lower.shape != (W.shape[1],)
        or upper.shape != lower.shape
        or not all(np.isfinite(a).all() for a in (W, b, v, lower, upper))
        or not np.isfinite(bias)
        or np.any(lower > upper)
    ):
        raise ValueError("Finite compatible network and box arrays required")
    return W, b, v, float(bias), lower, upper


def evaluate(net, x):
    W, b, v, bias, _lo, _hi = net
    x = np.asarray(x, dtype=float)
    return np.maximum(x @ W.T + b, 0) @ v + bias


def maximize(net, threshold=0.0, time_limit=10.0, tolerance=1e-7):
    W, b, v, bias, lo, hi = network(*net)
    d, m = W.shape[1], W.shape[0]
    if (
        not np.isfinite(threshold)
        or not np.isfinite(time_limit)
        or time_limit <= 0
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("Invalid limits")
    low = np.maximum(W, 0) @ lo + np.minimum(W, 0) @ hi + b
    high = np.maximum(W, 0) @ hi + np.minimum(W, 0) @ lo + b
    # x, h=ReLU(Wx+b), a=activation bits. Interval-derived big-M, no guessed constant.
    lb = np.r_[lo, np.zeros(m), np.zeros(m)]
    ub = np.r_[hi, np.maximum(high, 0), np.ones(m)]
    for j in range(m):
        if low[j] >= 0:
            lb[d + m + j] = 1
        elif high[j] <= 0:
            ub[d + m + j] = 0
    rows = []
    limits = []
    for j in range(m):
        row = np.zeros(d + 2 * m)
        row[:d] = W[j]
        row[d + j] = -1
        rows.append(row)
        limits.append(-b[j])
        row = np.zeros(d + 2 * m)
        row[d + j] = 1
        row[d + m + j] = -max(high[j], 0)
        rows.append(row)
        limits.append(0)
        row = np.zeros(d + 2 * m)
        row[:d] = -W[j]
        row[d + j] = 1
        row[d + m + j] = -min(low[j], 0)
        rows.append(row)
        limits.append(b[j] - min(low[j], 0))
    c = np.r_[np.zeros(d), -v, np.zeros(m)]
    r = milp(
        c,
        integrality=np.r_[np.zeros(d + m), np.ones(m)],
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(np.array(rows), -np.inf, limits),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    witness = None if r.x is None else r.x[:d]
    value = None if witness is None else float(evaluate(net, witness))
    if witness is not None and (
        np.any(witness < lo - 1e-6)
        or np.any(witness > hi + 1e-6)
        or abs(value - (bias - r.fun)) > 1e-5
    ):
        raise RuntimeError("Independent witness audit failed")
    dual = getattr(r, "mip_dual_bound", None)
    bound = None if dual is None or not np.isfinite(dual) else float(bias - dual)
    if value is not None and value > threshold + tolerance:
        status = "counterexample"
    elif bound is not None and bound <= threshold + tolerance:
        status = "certified_with_tolerance"
    else:
        status = "unknown"
    return {
        "status": status,
        "solver_status": int(r.status),
        "maximum_witness": value,
        "upper_bound": bound,
        "threshold": threshold,
        "tolerance": tolerance,
        "witness": None if witness is None else witness.tolist(),
    }


def enumerate_regions(net):
    """Independent oracle: one continuous LP for each activation pattern."""
    W, b, v, bias, lo, hi = network(*net)
    m = len(b)
    if m > 14:
        raise ValueError("Enumeration capped at 14 hidden neurons")
    best = -np.inf
    for bits in itertools.product((0, 1), repeat=m):
        a = np.array(bits)
        sign = 1 - 2 * a
        A = sign[:, None] * W
        rhs = -sign * b
        c = -(v * a) @ W
        r = linprog(c, A_ub=A, b_ub=rhs, bounds=list(zip(lo, hi, strict=False)), method="highs")
        if r.success:
            best = max(best, float(-r.fun + (v * a) @ b + bias))
    return best


def benchmark():
    # Endpoints are harmless; the interior triangular bump violates 0.4.
    net = network([[1], [1], [1]], [0, -0.5, -1], [1, -2, 1], 0, [0], [1])
    return {
        "scope": "one hidden ReLU layer; floating-point solver certificate, not formal arithmetic",
        "endpoint_values": evaluate(net, [[0], [1]]).tolist(),
        "region_oracle": enumerate_regions(net),
        "unsafe_limit": maximize(net, 0.4),
        "safe_limit": maximize(net, 0.6),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results.json")
    args = p.parse_args()
    Path(args.output).write_text(json.dumps(benchmark(), indent=2))
