"""Model-agnostic attribution primitives for plan_updated.md section 9.

Section 9 asks for one comparable object across six very different models: a
vector of "share of contribution per variable group", summing to one.  Each
adapter's attributions are also available per feature (``importance_by_column``),
so a group share can be split into the measurements behind it -- temperature vs.
irradiance vs. cloud cover inside the weather block.  Each
model exposes attribution through a different mechanism, so the shared layer is
the *post-processing*: group mapping, normalisation, agreement matrix, and the
cross-check against the L0-L5 ablation.

Model adapters live in ``scripts/xai_report.py``; nothing here imports a model
library, so it stays testable without a GPU or a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb, factorial
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.exogenous.feature_store import (
    FEATURE_GROUPS,
    is_categorical_column,
    xai_feature_groups,
)

GROUP_ORDER = ("hist_d1", "hist_d2", "hist_d3", "hist_d4",
               "hour", "day_of_week", "month", "calendar", "weather",
               "price", "neighbor", "state", "metadata", "other")

#: Players of the target's own past, most recent first.  These are what
#: TimeSHAP calls the event axis and what section 9 previously left out; the
#: target history carries 87-89 % of the mass in Hertel et al. (E-Energy '26,
#: Table 3), so a share vector without them is not comparable to the literature.
TARGET_PLAYERS = ("hist_d1", "hist_d2", "hist_d3", "hist_d4")

IDENTIFIER_COLUMNS = ("unique_id", "ds", "y")


def column_groups(columns: Iterable[str]) -> Dict[str, str]:
    """Map each column to its group, falling back to the shared feature-store rule."""
    return xai_feature_groups(list(columns))


def importance_by_group(importances: Mapping[str, float],
                        groups: Optional[Mapping[str, str]] = None) -> pd.Series:
    """Absolute contribution per group, normalised to sum to one.

    Adapters pass magnitudes where the method gives them (mean |SHAP|, MAE
    increase); signed vectors (raw TreeSHAP) are summed with their sign first and
    taken in absolute value per group, so a group whose positive and negative
    effects cancel is not counted twice.  The result is the "share of
    contribution" vector of section 9; a model whose attributions are all zero
    returns an even split so the Spearman matrix stays defined, and the caller
    logs the degenerate case.
    """
    columns = list(importances.keys())
    if groups is None:
        groups = column_groups(columns)
    totals: Dict[str, float] = {}
    for column, value in importances.items():
        group = groups.get(column, "other")
        try:
            totals[group] = totals.get(group, 0.0) + float(value)
        except (TypeError, ValueError):
            continue
    ordered = [group for group in GROUP_ORDER if group in totals]
    ordered += [group for group in totals if group not in GROUP_ORDER]
    vector = pd.Series({group: abs(totals[group]) for group in ordered}, dtype=float)
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0.0:
        if len(vector) == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(vector), index=vector.index)
    return vector / total


def importance_by_column(importances: Mapping[str, float],
                         columns: Optional[Sequence[str]] = None) -> pd.Series:
    """Absolute contribution per feature column, normalised to sum to one.

    :func:`importance_by_group` collapses e.g. the whole weather block into one
    number; this keeps every column, so the reader sees whether the weather share
    comes from temperature, irradiance, cloud cover, or wind.  Magnitudes only:
    a signed vector is taken in absolute value per column, the same definition
    the group vector uses.  ``columns`` fixes the index (and keeps a column the
    model never used at 0.0) so rows stay comparable column-by-column.
    """
    ordered = list(columns) if columns is not None else list(importances.keys())
    values: Dict[str, float] = {}
    for column in ordered:
        try:
            values[column] = abs(float(importances.get(column, 0.0)))
        except (TypeError, ValueError):
            values[column] = 0.0
    vector = pd.Series(values, dtype=float)
    if len(vector) == 0:
        return vector
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0.0:
        return pd.Series(1.0 / len(vector), index=vector.index)
    return vector / total


def column_shares_frame(per_model: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Models x features share matrix; columns keep the order the models saw them."""
    vectors = {name: importance_by_column(values) for name, values in per_model.items()}
    frame = pd.DataFrame(vectors).T.fillna(0.0)
    order: List[str] = []
    for values in per_model.values():
        for column in values:
            if column not in order:
                order.append(column)
    return frame[[column for column in order if column in frame.columns]]


def shares_frame(per_model: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Models x groups share matrix, groups in canonical order."""
    vectors = {name: importance_by_group(values) for name, values in per_model.items()}
    frame = pd.DataFrame(vectors).T.fillna(0.0)
    ordered = [group for group in GROUP_ORDER if group in frame.columns]
    ordered += [group for group in frame.columns if group not in GROUP_ORDER]
    return frame[ordered]


def spearman_matrix(shares: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between the group-share vectors of the models."""
    if shares.empty:
        return pd.DataFrame()
    return shares.T.corr(method="spearman").fillna(np.nan)


def _mae(prediction, truth) -> float:
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.mean(np.abs(prediction - truth)))


def tree_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values of a LightGBM regressor (``pred_contrib`` is exact TreeSHAP).

    Returns one column per feature plus ``bias``; the bias is dropped by the
    callers before grouping because it is not a contribution of any variable.
    """
    contributions = model.predict(X, pred_contrib=True)
    contributions = np.asarray(contributions, dtype=float)
    columns = list(X.columns) + ["bias"]
    return pd.DataFrame(contributions, columns=columns, index=X.index)


def block_permute(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle whole blocks of a series, keeping within-block order.

    A plain element-wise permutation destroys the daily shape and makes any
    feature look important; permuting whole days keeps the diurnal profile and
    only breaks the alignment with the target.
    """
    values = np.asarray(values)
    n = len(values)
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    order = rng.permutation(n_blocks)
    chunks = [values[i * block:(i + 1) * block] for i in order]
    return np.concatenate(chunks)[:n]


def permutation_importance(predict: Callable[[pd.DataFrame], np.ndarray],
                           frame: pd.DataFrame, truth,
                           columns: Sequence[str],
                           block: int = 96, n_repeats: int = 5,
                           seed: int = 0) -> Dict[str, float]:
    """Increase in MAE when one covariate's daily blocks are shuffled.

    ``predict`` must accept a full frame and return the forecast for ``truth``.
    """
    baseline = _mae(predict(frame), truth)
    rng = np.random.default_rng(seed)
    importances: Dict[str, float] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        deltas = []
        for _ in range(int(n_repeats)):
            perturbed = frame.copy()
            perturbed[column] = block_permute(frame[column].to_numpy(), block, rng)
            deltas.append(_mae(predict(perturbed), truth) - baseline)
        importances[column] = float(np.mean(deltas))
    return importances


def occlusion_importance(predict: Callable[[pd.DataFrame], np.ndarray],
                         frame: pd.DataFrame, truth,
                         windows: Sequence[tuple],
                         groups: Mapping[str, Sequence[str]],
                         neutral: str = "mean") -> Dict[str, float]:
    """MAE increase when a group is blanked inside one day-window at a time.

    ``windows`` is ``[(label, positions)]`` with positional row indices to
    occlude; ``groups`` maps a variable group to its columns.  This is the shared
    implementation behind PatchTST/TFT occlusion and the covariate ablation of
    the two TSFM models (section 9).
    """
    baseline = _mae(predict(frame), truth)
    importances: Dict[str, float] = {}
    for group, columns in groups.items():
        usable = [column for column in columns if column in frame.columns]
        if not usable:
            continue
        deltas = []
        for _, positions in windows:
            labels = frame.index[list(positions)]
            perturbed = frame.copy()
            for column in usable:
                values = frame[column]
                fill = values.mean() if neutral == "mean" else 0.0
                perturbed.loc[labels, column] = fill
            deltas.append(_mae(predict(perturbed), truth) - baseline)
        importances[group] = float(np.mean(deltas))
    return importances


def day_windows(index, steps_per_day: int, last_n_days: Optional[int] = None,
                step: Optional[int] = None) -> List[tuple]:
    """Positional windows of one day over ``index``, oldest first."""
    n = len(index)
    if n == 0:
        return []
    block = max(1, int(steps_per_day))
    count = n // block
    if count == 0:
        return [("day_0", np.arange(n))]
    windows = []
    for day in range(max(0, count - (last_n_days or count))):
        start = day * block
        windows.append((f"day_{day}", np.arange(start, start + block)))
    if step and step > 1:
        windows = windows[::step]
    return windows


def compare_with_ablation(shares: pd.DataFrame, ablation: pd.DataFrame) -> pd.DataFrame:
    """Join group shares with the L0-L5 effect of the same group.

    ``ablation`` must have ``group`` and ``delta_mae`` columns (negative means the
    group helped) and may carry a ``model`` column, which is how the elimination
    ledger arrives here.  A group with a high share but no measurable benefit is
    exactly the contradiction section 9 says must be explained, so it is flagged.
    """
    if shares.empty:
        return pd.DataFrame(columns=["model", "group", "share", "delta_mae", "flag"])
    long = shares.reset_index().melt(id_vars="index", var_name="group", value_name="share")
    long = long.rename(columns={"index": "model"})
    if ablation is None or ablation.empty:
        long["delta_mae"] = np.nan
        long["flag"] = ""
        return long
    # an elimination ledger carries one row per (model, group), the old ladder
    # table one per group; merge on whatever the source actually keys on
    keys = ["group"]
    if "model" in ablation.columns:
        keys = ["model", "group"]
    merged = long.merge(ablation[keys + ["delta_mae"]], on=keys, how="left")
    high_share = merged["share"] >= 0.10
    no_gain = merged["delta_mae"].notna() & (merged["delta_mae"] >= 0.0)
    merged["flag"] = np.where(high_share & no_gain,
                              "high attribution, no ablation gain - explain", "")
    return merged


def plot_group_shares(shares: pd.DataFrame, path, title: str = "Share of contribution by group"):
    """Stacked bar chart of the share vectors, one bar per model."""
    import matplotlib.pyplot as plt

    if shares.empty:
        return None
    figure, axis = plt.subplots(figsize=(max(6.0, 1.2 * len(shares)), 3.6))
    shares.plot(kind="bar", stacked=True, ax=axis, colormap="tab20", width=0.75)
    axis.set_ylabel("share of |contribution|")
    axis.set_title(title)
    axis.set_xlabel("")
    axis.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7, frameon=False)
    figure.tight_layout()
    path = str(path)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_spearman_heatmap(matrix: pd.DataFrame, path, title: str = "Spearman agreement"):
    """Heatmap of the pairwise Spearman correlations of the share vectors."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if matrix.empty:
        return None
    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=-1.0, vmax=1.0,
                square=True, cbar_kws={"label": "Spearman rho"}, ax=axis)
    axis.set_title(title)
    figure.tight_layout()
    path = str(path)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# Shapley game over a forecast window (plan_updated.md section 9, docs/
# timeshap_xai_de_xuat.md section 3).  The players are the last four days of the
# target plus one player per covariate group; the payoff is the whole horizon, so
# one coalition returns every forecast step and phi comes out per step.
# --------------------------------------------------------------------------


@dataclass(eq=False)
class Player:
    """One member of the game: a block of target rows, or a set of columns.

    ``rows`` marks a temporal player (positional rows of ``y`` inside the window,
    blanked as one coalition); ``columns`` marks a covariate player.  A player is
    one kind or the other, never both, so ``temporal`` tells the perturbation
    path apart.
    """

    name: str
    group: str
    rows: Optional[np.ndarray] = None
    columns: Tuple[str, ...] = ()

    @property
    def temporal(self) -> bool:
        return self.rows is not None


def temporal_players(n_points: int, steps_per_day: int, n_days: int = 4) -> List[Tuple[str, np.ndarray]]:
    """``[(name, positions)]`` of the target's past, most recent day first.

    Days are counted back from the end of the window, so ``hist_d1`` is the day
    immediately before the forecast origin.  The oldest player absorbs any
    remainder, which keeps every row of the window inside exactly one player: a
    row that belongs to no player is a row whose removal is never measured.
    """
    block = max(1, int(steps_per_day))
    n_points = int(n_points)
    if n_points <= 0:
        return []
    days = max(1, int(n_days))
    players: List[Tuple[str, np.ndarray]] = []
    end = n_points
    for day in range(1, days + 1):
        if end <= 0:
            break
        start = 0 if day == days else max(0, end - block)
        if start == end:
            break
        players.append((f"hist_d{day}", np.arange(start, end)))
        end = start
    return players


def periodic_background(frame: pd.DataFrame, steps_per_day: int,
                        columns: Optional[Sequence[str]] = None,
                        train_end: Optional[int] = None) -> pd.DataFrame:
    """Background matrix ``B``: the mean of each column at the same clock slot.

    TimeSHAP's ``B`` is the average feature value on the training set.  A load
    series is periodic, so the average has to be taken per slot of the day -- one
    number per column (``utils/xai.occlusion_importance``'s scalar fill) turns a
    diurnal covariate into a flat line, which is a much larger perturbation than
    switching the player off and inflates every attribution it touches.
    """
    block = max(1, int(steps_per_day))
    if frame.empty:
        raise ValueError("empty frame has no background")
    if columns is None:
        columns = [c for c in frame.columns if c != "unique_id"]
    columns = [c for c in columns if c in frame.columns]
    if not columns:
        raise ValueError("no column to build a background from")
    train = frame if train_end is None else frame.iloc[:int(train_end)]
    if train.empty:
        raise ValueError("training slice of the background is empty")
    slots = np.arange(len(train)) % block
    means = train[columns].groupby(slots).mean().reindex(range(block)).ffill().bfill()
    positions = np.arange(len(frame)) % block
    return pd.DataFrame(means.to_numpy(dtype=float)[positions],
                        index=frame.index, columns=columns)


def players_for_frame(frame: pd.DataFrame, columns: Optional[Sequence[str]] = None,
                      steps_per_day: int = 96, n_days: int = 4,
                      granularity: str = "group",
                      include_target: bool = True) -> List[Player]:
    """The player set of one window: four days of target plus the covariate groups."""
    players: List[Player] = []
    if include_target:
        players += [Player(name=name, group=name, rows=positions)
                    for name, positions in temporal_players(len(frame), steps_per_day, n_days)]
    usable = [c for c in (frame.columns if columns is None else columns)
              if c not in IDENTIFIER_COLUMNS and c in frame.columns]
    if granularity == "column":
        for column in usable:
            players.append(Player(name=column,
                                  group=column_groups([column]).get(column, "other"),
                                  columns=(column,)))
        return players
    if granularity != "group":
        raise ValueError(f"unknown granularity {granularity!r}; use 'group' or 'column'")
    buckets: Dict[str, List[str]] = {}
    for column, group in column_groups(usable).items():
        buckets.setdefault(group, []).append(column)
    ordered = [g for g in GROUP_ORDER if g in buckets]
    ordered += [g for g in buckets if g not in GROUP_ORDER]
    for group in ordered:
        players.append(Player(name=group, group=group, columns=tuple(buckets[group])))
    return players


def coalition_frame(frame: pd.DataFrame, active: Iterable[str], players: Sequence[Player],
                    background: Optional[pd.DataFrame] = None,
                    mode: str = "background") -> pd.DataFrame:
    """``h_X(z)``: keep the active players, blank the rest.

    ``mode="background"`` replaces a dropped player with ``B`` (TimeSHAP equation
    6/7), ``"mask"`` sets it to NaN (Hertel et al.'s masking), ``"drop"`` deletes
    the covariate columns outright (their covariate masking; only valid when the
    model accepts a missing channel).  Temporal players are never dropped by
    "drop": a shorter context is a different window, not a coalition.
    """
    active = set(active)
    out = frame.copy()
    if mode not in ("background", "mask", "drop"):
        raise ValueError(f"unknown mode {mode!r}")
    for player in players:
        if player.name in active:
            continue
        if player.temporal:
            rows = player.rows
            if mode == "drop":
                raise ValueError("a temporal player cannot be dropped, only blanked")
            column = out.columns.get_loc("y")
            if mode == "mask":
                out.iloc[rows, column] = np.nan
            else:
                if background is None or "y" not in background.columns:
                    raise ValueError("background mode needs a 'y' column in B")
                out.iloc[rows, column] = background["y"].to_numpy(dtype=float)[rows]
            continue
        present = [c for c in player.columns if c in out.columns]
        if not present:
            continue
        if mode == "drop":
            out = out.drop(columns=present)
        elif mode == "mask":
            out[present] = np.nan
        else:
            if background is None:
                raise ValueError("background mode needs B")
            for column in present:
                if column not in background.columns:
                    raise ValueError(f"background has no column {column!r}")
                out[column] = background[column].to_numpy(dtype=float)
    return out


@dataclass
class ShapleyResult:
    """Signed attributions of one window: ``phi`` is (players x horizon)."""

    names: Tuple[str, ...]
    phi: np.ndarray
    base: np.ndarray
    evaluated: int
    exact: bool

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.phi, index=list(self.names))

    def shares(self) -> pd.Series:
        return player_shares(self.phi, self.names)


def player_shares(phi, names: Sequence[str]) -> pd.Series:
    """Share of |contribution| per player, summed over the horizon, total one.

    Same definition as ``importance_by_group``: magnitudes only, because a signed
    sum inside one player cancels exactly the drivers that push the forecast both
    ways during a day.  Contrast with ``comparison`` in Hertel et al. Table 3,
    which is also a percentage of absolute SHAP values.
    """
    magnitudes = np.abs(np.asarray(phi, dtype=float)).sum(axis=1)
    total = float(magnitudes.sum())
    if not np.isfinite(total) or total <= 0.0:
        return pd.Series(np.full(len(names), 1.0 / max(1, len(names))), index=list(names))
    return pd.Series(magnitudes / total, index=list(names))


def collapse_to_groups(shares: pd.Series, players: Sequence[Player]) -> pd.Series:
    """Sum column-level players into their group (no-op at group granularity)."""
    groups = {p.name: p.group for p in players}
    totals: Dict[str, float] = {}
    for name, value in shares.items():
        group = groups.get(str(name), str(name))
        totals[group] = totals.get(group, 0.0) + float(value)
    ordered = [g for g in GROUP_ORDER if g in totals] + [g for g in totals if g not in GROUP_ORDER]
    vector = pd.Series({g: totals[g] for g in ordered}, dtype=float)
    total = float(vector.sum())
    return vector / total if total > 0 else vector


def _coalition_values(payoff: Callable[[Tuple[str, ...]], np.ndarray],
                      names: Sequence[str]) -> np.ndarray:
    values = []
    for mask in range(1 << len(names)):
        active = tuple(name for i, name in enumerate(names) if mask >> i & 1)
        values.append(np.asarray(payoff(active), dtype=float).ravel())
    values = np.stack(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("a coalition produced a non-finite forecast; check the "
                         "background fill before attributing anything")
    return values


def shapley_exact(payoff: Callable[[Tuple[str, ...]], np.ndarray],
                  players: Sequence, max_players: int = 12) -> ShapleyResult:
    """Exact Shapley values from all ``2**N`` coalitions.

    Exact means the definition, not an approximation: with every coalition on the
    table the value is enumerated directly, so additivity holds to machine
    precision and the result carries no sampling variance at all.  ``N <= 12``
    caps the cost at 4096 forecasts, which is the size of the game here (4 days +
    at most 8 covariate groups).
    """
    names = tuple(p.name if isinstance(p, Player) else str(p) for p in players)
    n = len(names)
    if n == 0:
        raise ValueError("a game needs at least one player")
    if n > max_players:
        raise ValueError(f"{n} players is above the exact limit {max_players}; "
                         "use shapley_sampled or prune the player set")
    values = _coalition_values(payoff, names)
    base = values[0].copy()
    counts = np.array([bin(mask).count("1") for mask in range(1 << n)])
    # lookup by coalition SIZE, not by mask: Shapley's weight of a coalition is
    # s!(n-s-1)!/n! and a player is never weighed against the full coalition
    weight_by_size = np.array([factorial(s) * factorial(n - s - 1) / factorial(n)
                               for s in range(n)])
    phi = np.empty((n, values.shape[1]), dtype=float)
    for i in range(n):
        bit = 1 << i
        rest = np.array([mask for mask in range(1 << n) if not mask & bit], dtype=int)
        phi[i] = (weight_by_size[counts[rest]][:, None]
                  * (values[rest | bit] - values[rest])).sum(axis=0)
    return ShapleyResult(names=names, phi=phi, base=base,
                         evaluated=1 << n, exact=True)


def shapley_sampled(payoff: Callable[[Tuple[str, ...]], np.ndarray],
                    players: Sequence, n_samples: int = 4096,
                    seed: int = 0) -> ShapleyResult:
    """KernelSHAP with sampled coalitions, for a player set too large to enumerate.

    Same estimator as TimeSHAP: coalitions are drawn with the KernelSHAP kernel
    weight, and the two anchors (all off, all on) are kept so the solution has the
    efficiency property exactly while the interior is approximated.  Use this only
    above 12 players -- below that ``shapley_exact`` is cheaper *and* variance-free.
    """
    names = tuple(p.name if isinstance(p, Player) else str(p) for p in players)
    n = len(names)
    if n < 2:
        raise ValueError("a sampled game needs at least two players")
    rng = np.random.default_rng(int(seed))
    sizes = rng.integers(1, n, size=int(n_samples))
    order = np.argsort(rng.random((int(n_samples), n)), axis=1)
    sampled = (order < sizes[:, None]).astype(int)
    masks = np.vstack([np.zeros((1, n), dtype=int), np.ones((1, n), dtype=int), sampled])
    masks = np.unique(masks, axis=0)
    values = np.stack([np.asarray(payoff(tuple(names[i] for i in np.flatnonzero(row))),
                                  dtype=float).ravel() for row in masks])
    if not np.all(np.isfinite(values)):
        raise ValueError("a coalition produced a non-finite forecast; check the "
                         "background fill before attributing anything")
    counts = masks.sum(axis=1)
    weights = np.empty(len(masks), dtype=float)
    inner = (counts > 0) & (counts < n)
    weights[inner] = (n - 1) / (np.array([comb(n, int(s)) for s in counts[inner]])
                                * counts[inner] * (n - counts[inner]))
    # the anchors carry the efficiency constraint, so they get an effectively
    # infinite weight instead of a finite one
    weights[~inner] = 1e6
    design = np.column_stack([np.ones(len(masks)), masks])
    root = np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(design * root, values * root, rcond=None)
    return ShapleyResult(names=names, phi=solution[1:], base=solution[0].copy(),
                         evaluated=int(len(masks)), exact=False)


def shapley(payoff: Callable[[Tuple[str, ...]], np.ndarray], players: Sequence,
            n_samples: int = 4096, seed: int = 0,
            max_players_exact: int = 12) -> ShapleyResult:
    """Exact below the limit, sampled above it -- never silently the wrong one."""
    n = len(players)
    if n <= max_players_exact:
        return shapley_exact(payoff, players, max_players=max_players_exact)
    return shapley_sampled(payoff, players, n_samples=n_samples, seed=seed)


def additivity_residual(phi, base, prediction) -> float:
    """``max_t |sum_i phi[i,t] + base[t] - f(X)[t]| / max_t |f(X)[t]|``.

    The hard gate of the whole attribution: an exact game must hit zero to machine
    precision, and a sampled one must stay small.  A number that fails this is not
    an explanation, it is arithmetic that did not close.
    """
    prediction = np.asarray(prediction, dtype=float).ravel()
    total = np.asarray(phi, dtype=float).sum(axis=0) + np.asarray(base, dtype=float).ravel()
    scale = float(np.max(np.abs(prediction))) or 1.0
    return float(np.max(np.abs(total - prediction)) / scale)
