"""Block bootstrap for the confidence interval of an MAE difference.

plan_updated.md section 8 asks for a CI of the MAE gap between two models, with
blocks of one day.  The point is that a 192-step forecast error series is not
i.i.d.: hours of the day repeat, so an i.i.d. bootstrap would report an interval
narrower than the data supports.  Resampling whole days keeps the within-day
autocorrelation intact and is the standard fix (moving-block bootstrap).

The DM test in ``utils/dm_test.py`` is kept as published in the original paper
and is *not* corrected here; this module is the second, complementary view.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from utils.protocol import BOOTSTRAP_BLOCK_DAYS

DEFAULT_BLOCK_DAYS = BOOTSTRAP_BLOCK_DAYS
DEFAULT_N_BOOT = 2000


def steps_per_day_from_index(index) -> Optional[int]:
    """Points per wall-clock day implied by an index (96 at 15 min, 48 at 30 min)."""
    try:
        index = pd.DatetimeIndex(index).sort_values()
    except (TypeError, ValueError):
        return None
    if len(index) < 2:
        return None
    deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
    positive = deltas[deltas > 0]
    if not positive.size:
        return None
    median_step = float(np.median(positive))
    if median_step <= 0:
        return None
    return max(1, int(round(86400.0 / median_step)))


def resolve_block(index_or_length, block: Optional[int] = None,
                  block_days: float = DEFAULT_BLOCK_DAYS) -> int:
    """Block length in points.

    An explicit ``block`` wins.  Otherwise the length comes from the index (one
    day by default); with a bare integer length and no index, the block falls
    back to 96 points, the paper's day at 100 % sampling.
    """
    if block is not None:
        return max(1, int(block))
    if not isinstance(index_or_length, (int, np.integer)):
        per_day = steps_per_day_from_index(index_or_length)
        if per_day is not None:
            return max(1, int(round(per_day * float(block_days))))
    return max(1, int(round(96 * float(block_days))))


def _circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a circular moving-block resample of length ``n``."""
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.reshape(-1)[:n]


def mean_difference_ci(
    differential,
    block: Optional[int] = None,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    block_days: float = DEFAULT_BLOCK_DAYS,
) -> Dict[str, float]:
    """Percentile CI of ``mean(differential)`` under a circular block bootstrap.

    ``differential`` is the per-step loss difference (loss_x - loss_y) or a
    Series with a datetime index.  The observed statistic is the plain mean; the
    interval says how much of the gap could be sampling noise.
    """
    if isinstance(differential, pd.Series):
        index = differential.index
        values = differential.dropna().to_numpy(dtype=float)
    else:
        index = None
        values = np.asarray(differential, dtype=float)
        values = values[~np.isnan(values)]
    n = len(values)
    if n < 5:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "block": float("nan"), "n_boot": 0.0, "n": float(n)}

    used_block = resolve_block(index if index is not None else n, block, block_days)
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = _circular_block_indices(n, used_block, rng)
        boot[b] = float(np.mean(values[idx]))
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    # Two-sided bootstrap p-value for H0: mean difference = 0.
    p_value = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
    return {
        "diff": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(max(p_value, 0.0), 1.0)),
        "block": float(used_block),
        "n_boot": float(n_boot),
        "n": float(n),
    }


def mae_difference_ci(
    loss_x,
    loss_y,
    block: Optional[int] = None,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    block_days: float = DEFAULT_BLOCK_DAYS,
) -> Dict[str, float]:
    """CI of mean(|e_x|) - mean(|e_y|) on the overlapping timestamps."""
    frame = pd.concat([pd.Series(loss_x, name="x"), pd.Series(loss_y, name="y")],
                      axis=1, join="inner").dropna()
    if frame.empty:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "block": float("nan"), "n_boot": 0.0, "n": 0.0,
                "mae_x": float("nan"), "mae_y": float("nan")}
    result = mean_difference_ci(frame["x"].to_numpy() - frame["y"].to_numpy(),
                                block=block, n_boot=n_boot, alpha=alpha, seed=seed,
                                block_days=block_days)
    result["mae_x"] = float(frame["x"].mean())
    result["mae_y"] = float(frame["y"].mean())
    return result


def pairwise_bootstrap_table(
    losses: Dict[str, Optional[pd.Series]],
    block: Optional[int] = None,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    model_order: Optional[Iterable[str]] = None,
    block_days: float = DEFAULT_BLOCK_DAYS,
) -> pd.DataFrame:
    """One row per ordered model pair, ready for the appendix table."""
    names = [name for name in (model_order or losses.keys()) if losses.get(name) is not None]
    rows = []
    for i, model_x in enumerate(names):
        for model_y in names[i + 1:]:
            result = mae_difference_ci(losses[model_x], losses[model_y], block=block,
                                       n_boot=n_boot, alpha=alpha, seed=seed,
                                       block_days=block_days)
            rows.append({
                "model_x": model_x, "model_y": model_y,
                "mae_x": result.get("mae_x"), "mae_y": result.get("mae_y"),
                "diff": result["diff"], "ci_low": result["ci_low"],
                "ci_high": result["ci_high"], "p_value": result["p_value"],
                "block": result["block"], "n": result["n"],
            })
    columns = ["model_x", "model_y", "mae_x", "mae_y", "diff", "ci_low",
               "ci_high", "p_value", "block", "n"]
    return pd.DataFrame(rows, columns=columns)
