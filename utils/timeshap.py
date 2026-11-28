"""TimeSHAP: one coalition game over target history and covariate blocks.

The design is the one of the proposal (``plan/timeshap_xai_de_xuat.md``):

* **Players** — four one-day players of the target history plus one player per
  covariate block.  Four days is not a choice: the context is ``2 x horizon``,
  which is exactly four days at every sampling rate, so the buckets are equal and
  none is empty.
* **Perturbation** — ``h_X(z) = X ⊙ D_z + B ⊙ (1 - D_z)``.  ``B`` is the
  *hour-of-day* mean of the training rows, not a single column mean: flattening a
  daily cycle into a constant would create a perturbation larger than "switching
  the variable off" and every attribution would be inflated.
* **Estimator** — with ``N <= 12`` players the game is enumerated exactly, so the
  values are marginal contributions, not a weighted regression: no sampling
  error, no seed, and additivity holds to machine precision.
* **Payoff** — the whole forecast vector, not one number, so the explanation can
  also say *when* a block matters without extra model calls.

Nothing here refits a model.  A game whose ``f`` changes between coalitions has
no Shapley values to speak of, so a model that cannot be re-fed a perturbed input
has to be handled by the caller (refit explicitly, or be left out).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.covariates import base_variable


@dataclass(frozen=True)
class Player:
    name: str
    positions: Tuple[int, ...]
    columns: Tuple[str, ...]
    kind: str = "covariate"

    @property
    def size(self) -> int:
        return len(self.positions)


def temporal_players(n_context: int, steps_per_day: int) -> List[Player]:
    """``hist_d1`` (most recent day) .. ``hist_d4`` (the rest of the context).

    The target history carries the largest part of the attribution in the
    literature (87-89 % in Hertel et al.), so a game without it is not comparable
    to anything and cannot answer "does the model use the fourth day".
    """
    players = []
    for day in range(1, 5):
        end = n_context - (day - 1) * steps_per_day
        start = max(0, n_context - day * steps_per_day)
        if end <= start:
            break
        players.append(Player(name=f"hist_d{day}", positions=tuple(range(start, end)),
                              columns=("y",), kind="temporal"))
    return players


def covariate_players(frame: pd.DataFrame, groups: Dict[str, Sequence[str]],
                      roles: Dict[str, Sequence[str]], n_context: int) -> List[Player]:
    """One player per covariate block.

    A known-future block owns its rows inside the horizon as well: switching the
    calendar off has to switch it off everywhere, otherwise the player is only
    half removed and the attribution is not a marginal contribution.  A
    past-only block (weather, state of charge) owns the context rows only, because
    that is all the model is ever given.
    """
    known = set(roles.get("known_future", []))
    players = []
    for name, columns in groups.items():
        present = tuple(c for c in columns if c in frame.columns)
        if not present:
            continue
        rows = tuple(range(len(frame))) if all(c in known for c in present) else tuple(range(n_context))
        players.append(Player(name=name, positions=rows, columns=present, kind="covariate"))
    return players


def players_per_variable(frame: pd.DataFrame, roles: Dict[str, Sequence[str]],
                         n_context: int) -> List[Player]:
    """One player per covariate *variable*, instead of one per physical block.

    ``covariate_players`` groups columns (``group_columns``) so a player is "the
    calendar" rather than "the hour code".  Splitting is the answer to the other
    question - which *single* column the forecast leans on - and it costs a much
    larger game: ``C`` columns need ``2^C`` coalitions where the grouped game
    needs ``2^blocks``.

    A variable that is carried as a pair of columns stays **one** player:
    ``hour_sin`` and ``hour_cos`` are exact functions of each other, so scoring
    them separately would only split their credit arbitrarily at the price of
    doubling the game (``utils.covariates.base_variable`` names the pair).

    Two caveats are inherent to the perturbation, not to this function:

    * Columns that are deterministic functions of each other are never switched
      off independently in a coalition, so no Shapley game can separate them: the
      credit between them is arbitrary and must not be read as evidence about
      either one.  Such a pair is merged into one player above.
    * The rows a column owns follow its role, exactly as in the block game: a
      known-future column owns the horizon rows too, a past-only column only the
      context.
    """
    known = set(roles.get("known_future", []))
    order, grouped = [], {}
    for column in frame.columns:
        if column in ("unique_id", "ds", "y"):
            continue
        variable = base_variable(column)
        if variable not in grouped:
            grouped[variable] = []
            order.append(variable)
        grouped[variable].append(column)
    players = []
    for variable in order:
        columns = tuple(grouped[variable])
        rows = (tuple(range(len(frame))) if all(c in known for c in columns)
                else tuple(range(n_context)))
        players.append(Player(name=variable, positions=rows, columns=columns, kind="covariate"))
    return players


def periodic_background(frame: pd.DataFrame, steps_per_day: int, train_end: int,
                        cycles: Dict[str, int] | None = None) -> pd.DataFrame:
    """Hour-of-day mean of every numeric column, computed on the training rows only.

    Identifier columns (``unique_id``, ``ds``) are not part of the game and are
    left out: averaging a string is meaningless, and a timestamp is not a value a
    coalition can switch off.

    ``cycles`` gives a column its own cycle length in rows; the default is one day.
    A weekly column needs ``7 * steps_per_day``, because its counterfactual has to
    be *half a week* later rather than half a day: averaging a weekly cycle inside a
    daily slot flattens it to the series mean, so ``dayofweek`` used to be switched
    off as "the constant Wednesday" instead of being rotated.

    A column that already equals its own per-phase mean - the usual case is a
    calendar code equal to its slot, ``hour == index % steps_per_day`` on a
    60-minute grid - is replaced by the same profile rotated half a cycle, so that
    switching it off is a counterfactual instead of a no-op.  Without the
    rotation such a player has ``phi ≡ 0`` exactly (the background reproduces the
    column) and the time-of-day effect silently disappears from the game.  The
    comparison is tolerant and the *exact* column is used as the phase table: a
    cyclic encoding (``hour_sin``) has per-phase means that differ from the column
    in the last bits, and an exact-equality test would leave it with a
    numerically-zero perturbation instead of a half-cycle shift.
    """
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    train = frame.iloc[:train_end][numeric]
    if train.empty:
        raise ValueError("no training rows to build the background from")
    index = np.arange(len(frame))
    periods = {column: max(int((cycles or {}).get(column, steps_per_day)), 1)
               for column in numeric}

    profile = pd.DataFrame(index=frame.index, columns=numeric, dtype=float)
    for period in sorted(set(periods.values())):
        columns = [c for c in numeric if periods[c] == period]
        slots = index % period
        means = train[columns].groupby(slots[: len(train)]).mean()
        column_means = train[columns].mean()
        table = pd.DataFrame(index=range(period), columns=columns, dtype=float)
        for phase in range(period):
            table.loc[phase] = means.loc[phase] if phase in means.index else column_means
        profile[columns] = table.iloc[slots].to_numpy()

    for column in numeric:
        period = periods[column]
        shift = period // 2
        if shift <= 0:
            continue
        original = frame[column].to_numpy(dtype=float)
        values = profile[column].to_numpy(dtype=float)
        if not _is_phase_profile(original, values):
            continue
        table = np.full(period, float(np.nanmean(values)) if values.size else 0.0)
        take = min(period, len(original))
        table[:take] = original[:take]
        slots = index % period
        profile[column] = table[(slots + shift) % period]
    return profile


def _is_phase_profile(original: np.ndarray, values: np.ndarray, atol: float = 1e-9) -> bool:
    """Does ``values`` reproduce ``original`` up to rounding of the per-phase mean?"""
    if original.shape != values.shape:
        return False
    scale = float(np.nanmax(np.abs(values))) if values.size else 0.0
    return bool(np.allclose(original, values, rtol=0.0, atol=atol * max(scale, 1.0)))


def apply_coalition(frame: pd.DataFrame, background: pd.DataFrame, players: Sequence[Player],
                    active: Iterable[str], mode: str = "background") -> pd.DataFrame:
    """``h_X(z)``: rows of inactive players are replaced by the background.

    ``mode="background"`` uses the hour-of-day profile; ``mode="mean"`` uses one
    constant per column (the coarser baseline the proposal asks to compare
    against); ``mode="zero"`` blanks to zero, which is only meaningful for the
    scaled covariates and is offered for completeness.
    """
    active = set(active)
    perturbed = frame.copy()
    for player in players:
        if player.name in active:
            continue
        rows = list(player.positions)
        columns = [c for c in player.columns if c in frame.columns and c in background.columns]
        if not columns or not rows:
            continue
        positions = np.asarray(rows)
        for column in columns:
            if mode == "background":
                values = background[column].to_numpy()[positions]
            elif mode == "mean":
                values = np.repeat(frame[column].to_numpy()[positions].mean(), len(positions))
            elif mode == "zero":
                values = np.zeros(len(positions))
            else:
                raise ValueError(f"unknown perturbation mode {mode!r}")
            # Integer code columns (calendar) must stay integers: a float hour is
            # not a category, and the models embed these columns.
            if pd.api.types.is_integer_dtype(frame[column].dtype):
                values = np.rint(values).astype(frame[column].dtype)
            perturbed.iloc[positions, perturbed.columns.get_loc(column)] = values
    return perturbed


def drop_noop_players(frame: pd.DataFrame, players: Sequence[Player],
                      background: pd.DataFrame) -> Tuple[List[Player], List[str]]:
    """Split players into ``(kept, dropped)`` by whether their perturbation bites.

    Under the hour-of-day background, a column that already equals its own
    per-slot mean is replaced by itself, so the payoff cannot move and ``phi`` is
    exactly zero.  The usual case is a calendar column that equals its slot
    (``hour == index % steps_per_day``: a 60-minute grid, e.g. 15-minute data at
    sampling 25 %, or a gapless 30-minute series at 50 %).  That is a property of
    the perturbation, not evidence that the model ignores the clock, and the
    player only doubles the game.
    """
    kept, dropped = [], []
    for player in players:
        positions = list(player.positions)
        columns = [c for c in player.columns if c in frame.columns and c in background.columns]
        if not positions or not columns:
            kept.append(player)
            continue
        same = all(
            np.array_equal(frame[column].to_numpy(dtype=float)[positions],
                           background[column].to_numpy(dtype=float)[positions])
            for column in columns)
        if same:
            dropped.append(player.name)
        else:
            kept.append(player)
    return kept, dropped


def enumerate_game(payoff: Callable[[Tuple[str, ...]], np.ndarray], players: Sequence[Player],
                   max_players: int = 12) -> Dict[frozenset, np.ndarray]:
    """Evaluate ``f`` on all ``2^N`` coalitions (``N <= max_players``)."""
    if len(players) > max_players:
        raise ValueError(f"{len(players)} players need {2 ** len(players)} coalitions; "
                         f"raise --max-players above {max_players} only if that cost is affordable, "
                         "group the covariates, or sample the game")
    names = [p.name for p in players]
    values: Dict[frozenset, np.ndarray] = {}
    for size in range(len(names) + 1):
        for subset in itertools.combinations(names, size):
            values[frozenset(subset)] = np.asarray(payoff(tuple(subset)), dtype=float)
    return values


def shapley_exact(values: Dict[frozenset, np.ndarray], players: Sequence[Player]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Shapley values of every player, per forecast step.

    With the full value table the marginal-contribution formula is available, so
    no kernel regression and no sampling is needed:
    ``phi_i = sum_S |S|!(N-|S|-1)!/N! (f(S+i) - f(S))``.
    """
    n = len(players)
    names = [p.name for p in players]
    horizon = next(iter(values.values())).shape[0]
    phi = np.zeros((n, horizon))
    empty = values[frozenset()]
    for index, name in enumerate(names):
        others = [other for other in names if other != name]
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in itertools.combinations(others, size):
                without = values[frozenset(subset)]
                with_player = values[frozenset(set(subset) | {name})]
                phi[index] += weight * (with_player - without)
    full = values[frozenset(names)]
    return phi, empty, full


def additivity_residual(phi: np.ndarray, base: np.ndarray, prediction: np.ndarray) -> float:
    """``max_t |sum_i phi[i, t] + base[t] - f(X)[t]| / max_t |f(X)[t]|``."""
    scale = float(np.max(np.abs(prediction))) or 1.0
    return float(np.max(np.abs(phi.sum(axis=0) + base - prediction)) / scale)


def shares_by_player(phi: np.ndarray, players: Sequence[Player]) -> pd.Series:
    """``sum_t |phi_i| / sum_i sum_t |phi_i|`` — shares that sum to one."""
    magnitude = np.abs(phi).sum(axis=1)
    total = float(magnitude.sum())
    values = magnitude / total if total > 0 else np.zeros_like(magnitude)
    return pd.Series(values, index=[p.name for p in players])


def group_shares(shares: pd.Series) -> pd.Series:
    """Collapse player shares to the reporting blocks of the table.

    The four history players become one ``target_history`` entry: the table asks
    which *block* carries the forecast, and the four days are a partition of one
    block.  The per-day detail stays in ``shap_values.csv``.

    The ``target_history`` row is emitted only when the game actually had history
    players.  A covariate-only game (``--covariate-only``), a model forced to
    ``include_temporal=False``, or an L0 run has no such player, and writing the
    seeded 0.0 would read as "the model ignores the past" when it means "the past
    was never a player" - the history is frozen into the base value instead.
    """
    grouped = {}
    history = 0.0
    has_history = False
    for name, value in shares.items():
        if name.startswith("hist_d"):
            history += float(value)
            has_history = True
        else:
            grouped[name] = float(value)
    if has_history:
        return pd.Series({"target_history": history, **grouped})
    return pd.Series(grouped)


def signed_by_player(phi: np.ndarray, players: Sequence[Player]) -> pd.Series:
    """Signed mean contribution per player, for the dependence-style table."""
    return pd.Series(phi.mean(axis=1), index=[p.name for p in players])


def temporal_pruning_report(phi: np.ndarray, players: Sequence[Player]) -> pd.DataFrame:
    """Per-day magnitude of the target-history players, in order.

    Algorithm 1 of Bento et al. prunes the event axis for long sequences.  Here
    the axis has exactly four players, so the pruning rule has nothing to save;
    what it is used for in the proposal — justifying that a coarser time split
    would be equivalent — is answered directly by these four numbers: if the
    oldest day's share is already negligible, a coarser split is admissible.
    """
    rows = []
    for index, player in enumerate(players):
        if player.kind != "temporal":
            continue
        rows.append({"player": player.name, "steps": player.size,
                     "mean_abs_contribution": float(np.abs(phi[index]).mean())})
    # Keep the header when the history players are held fixed (covariate-only game).
    return pd.DataFrame(rows, columns=["player", "steps", "mean_abs_contribution"])
