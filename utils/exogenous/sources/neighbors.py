"""Neighbour (L5) source: a time x entity matrix of peer load series.

Selection matters as much as the lag: choosing "similar households" using the
test window would leak the answer.  Similarity here is therefore computed only
on rows at or before ``selection_end`` (the forecast origin of the test window).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from utils.exogenous.config import NEIGHBOR_DIR, NEIGHBOR_SOURCES


def matrix_path(dataset_name):
    filename = NEIGHBOR_SOURCES.get(str(dataset_name).lower())
    if filename is None:
        return None
    return Path(NEIGHBOR_DIR) / filename


def load_matrix(dataset_name):
    """Load the neighbour matrix (index = UTC timestamps, columns = entity ids)."""
    path = matrix_path(dataset_name)
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame.sort_index()


def select_neighbors(matrix, target_entity, k, selection_end, min_coverage=0.8):
    """Pick the k most similar peers using data up to ``selection_end`` only.

    Similarity is the Pearson correlation of the average daily profile plus a
    penalty for a different consumption level, computed on the hourly mean of the
    pre-origin window.  A peer with less than ``min_coverage`` of the window is
    excluded instead of being imputed.
    """
    if matrix is None or matrix.empty:
        return []

    candidates = [c for c in matrix.columns if str(c) != str(target_entity)]
    if not candidates:
        return []

    window = matrix.loc[matrix.index <= pd.Timestamp(selection_end), candidates]
    if window.empty:
        return []

    hourly = window.resample("1h").mean()
    coverage = hourly.notna().mean()
    eligible = [c for c in candidates if coverage.get(c, 0.0) >= min_coverage]
    if not eligible:
        return []

    if str(target_entity) not in matrix.columns:
        # Single-house datasets: rank by level, which leaks no future information.
        hourly = matrix.loc[matrix.index <= pd.Timestamp(selection_end), candidates].resample("1h").mean()
        coverage = hourly.notna().mean()
        eligible = [c for c in candidates if coverage.get(c, 0.0) >= min_coverage]
        if not eligible:
            return []
        level = hourly[eligible].mean()
        return list(level.sort_values().index[:k])

    target_series = matrix.loc[matrix.index <= pd.Timestamp(selection_end), str(target_entity)]

    target_hourly = target_series.resample("1h").mean()
    target_profile = _daily_profile(target_hourly)

    scored = []
    for column in eligible:
        profile = _daily_profile(hourly[column])
        corr = _safe_corr(target_profile, profile)
        level_penalty = abs(hourly[column].mean() - target_hourly.mean()) / (abs(target_hourly.mean()) + 1e-9)
        scored.append((corr - 0.1 * level_penalty, column))

    scored.sort(reverse=True)
    return [column for _, column in scored[:k]]


def _daily_profile(hourly_series):
    prof = hourly_series.groupby(hourly_series.index.hour).mean()
    return prof.reindex(range(24))


def _safe_corr(a, b):
    mask = a.notna() & b.notna()
    if mask.sum() < 6:
        return -1.0
    x = a[mask].to_numpy(dtype=float)
    y = b[mask].to_numpy(dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return -1.0
    return float(np.corrcoef(x, y)[0, 1])


def aggregate(matrix, columns):
    """Equal-weight mean of the chosen peers, on the matrix index."""
    if matrix is None or not columns:
        return None
    return matrix[columns].mean(axis=1)


def aligned_aggregate(dataset_name, target_entity, timestamps, selection_end, k=8):
    """Load, select and aggregate in one call.

    Returns ``(series, info)`` where info records the chosen peers and whether the
    target itself was part of the matrix.
    """
    matrix = load_matrix(dataset_name)
    if matrix is None:
        return None, {"reason": f"no neighbour matrix at {matrix_path(dataset_name)}"}
    chosen = select_neighbors(matrix, target_entity, k, selection_end)
    if not chosen:
        return None, {"reason": "no eligible peers in the pre-origin window"}
    aggregate_series = aggregate(matrix, chosen).reindex(pd.DatetimeIndex(timestamps))
    return aggregate_series, {"neighbors": chosen, "n": len(chosen)}
