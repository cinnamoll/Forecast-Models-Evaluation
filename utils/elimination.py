"""Backward elimination of covariates driven by TimeSHAP shares.

The question this answers is not "which variables matter" (that is section 9's
share vector) but "which variables can be removed without paying for it": start
from every covariate, measure the game, drop the player with the smallest share,
measure again, and stop when the next drop costs more MAE than the run-to-run
noise allows.  Attributions are recomputed after every drop on purpose -- a group
that is important next to the weather can be irrelevant once the weather is gone,
and a one-shot ranking cannot see that.

Two rules are not negotiable, because the alternative produces a number that only
looks like an improvement:

* **Selection happens on the validation block, never on the test window.**  If the
  shares that decide what to drop are computed on the same 192 points that are
  later reported, the reported MAE is the result of a choice made with knowledge
  of the answer.  ``selection_mae`` is therefore measured on the block of
  ``protocol.validation_windows`` and only ``test_mae`` is a reportable number.
* **A drop is accepted only if it stays inside the noise.**  ``tolerance`` is the
  MAE difference a run-to-run comparison can produce by itself; with seeds taken
  from the clock (plan section 0) that is the IQR of the L0 runs, not zero.  A
  drop that buys 0.0004 MAE while the seed spread is 0.01 is not an improvement,
  it is a reseeding.

Nothing here imports a model: ``eliminate`` takes a ``measure`` callable, so the
loop is testable without a GPU and the model-specific wiring stays in
``scripts/xai_elimination.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.xai import TARGET_PLAYERS, Player


@dataclass
class Measurement:
    """Attributions and scores of one column set.

    ``shares`` is per player (one per group, or one per column) and ``groups`` is
    the same measurement collapsed to groups; both sum to one.  ``additivity`` is
    the residual of ``additivity_residual`` and is kept next to the scores because
    an explanation that does not close cannot justify a drop.
    """

    columns: Tuple[str, ...]
    shares: pd.Series
    groups: pd.Series
    selection_mae: float
    test_mae: float
    players: Tuple[Player, ...]
    evaluated: int = 0
    exact: bool = True
    additivity: float = 0.0
    notes: Dict[str, object] = field(default_factory=dict)


@dataclass
class EliminationResult:
    rounds: List[Measurement]
    dropped: List[str]
    kept: Tuple[str, ...]
    stop_reason: str
    best_index: int

    @property
    def best(self) -> Measurement:
        return self.rounds[self.best_index]

    def ledger(self) -> pd.DataFrame:
        rows = []
        for index, measure in enumerate(self.rounds):
            dropped = self.dropped[index] if index < len(self.dropped) else ""
            rows.append({
                "round": index,
                "n_columns": len(measure.columns),
                "dropped_next": dropped,
                "columns": " ".join(measure.columns),
                "selection_mae": measure.selection_mae,
                "test_mae": measure.test_mae,
                "additivity": measure.additivity,
                "evaluated": measure.evaluated,
                "exact": measure.exact,
                "kept": " ".join(self.kept) == " ".join(measure.columns),
            })
        return pd.DataFrame(rows)

    def share_ledger(self) -> pd.DataFrame:
        rows = []
        for index, measure in enumerate(self.rounds):
            for name, value in measure.groups.items():
                rows.append({"round": index, "group": name, "share": float(value)})
        return pd.DataFrame(rows, columns=["round", "group", "share"])


def weakest_covariate(measure: Measurement,
                      protect: Sequence[str] = TARGET_PLAYERS,
                      exclude: Sequence[str] = ()) -> Optional[Player]:
    """The covariate player with the smallest share, or ``None``.

    Temporal players are never candidates: dropping the target's own past is not a
    covariate-selection decision, it is a different forecasting problem, and a
    model without history is not "the same model with fewer variables".
    """
    blocked = set(protect) | set(exclude)
    candidates = [p for p in measure.players if p.name not in blocked and not p.temporal]
    if not candidates:
        return None
    ranked = sorted(candidates,
                    key=lambda p: (float(measure.shares.get(p.name, 0.0)), p.name))
    return ranked[0]


def columns_without(player: Player, columns: Sequence[str]) -> Tuple[str, ...]:
    """Column set after removing one player (a group, or a single column)."""
    removed = set(player.columns)
    return tuple(c for c in columns if c not in removed)


def eliminate(measure: Callable[[Tuple[str, ...]], Measurement],
              columns: Sequence[str],
              tolerance: float = 0.0,
              max_rounds: Optional[int] = None,
              min_covariates: int = 1,
              protect: Sequence[str] = TARGET_PLAYERS,
              log: Optional[Callable[[str], None]] = None) -> EliminationResult:
    """Drop the weakest covariate group until a drop is too expensive.

    ``measure(columns)`` fits the model on exactly those columns and returns a
    :class:`Measurement`; everything else is bookkeeping.  ``tolerance`` is how
    much selection MAE a drop may add before it is refused -- set it to the IQR of
    the model's L0 runs, because anything smaller than the seed spread is noise
    wearing a delta's clothes.
    """
    say = log or (lambda message: None)
    columns = tuple(columns)
    if not columns:
        raise ValueError("elimination needs at least one covariate to start from")

    first = measure(columns)
    rounds: List[Measurement] = [first]
    dropped: List[str] = []
    limit = len([p for p in first.players if not p.temporal]) - int(min_covariates)
    if max_rounds is not None:
        limit = min(limit, int(max_rounds))
    say(f"round 0: {len(columns)} columns, {len(first.shares)} players, "
        f"selection MAE {first.selection_mae:.6f}, test MAE {first.test_mae:.6f}")

    reason = "no covariate player left"
    current = first
    while True:
        if len(dropped) >= limit:
            reason = (f"round budget reached ({len(dropped)} drops); "
                      "the curve is not necessarily flat yet")
            break
        candidate = weakest_covariate(current, protect=protect)
        if candidate is None:
            reason = "no covariate player left to drop"
            break
        trial_columns = columns_without(candidate, current.columns)
        if not trial_columns:
            reason = "dropping the weakest player would leave no covariate at all"
            break
        trial = measure(trial_columns)
        delta = float(trial.selection_mae - current.selection_mae)
        if delta > float(tolerance):
            reason = (f"dropping {candidate.name!r} costs {delta:+.6f} selection MAE, "
                      f"above the tolerance {float(tolerance):.6f}")
            say(f"refused: drop {candidate.name} costs {delta:+.6f}")
            break
        if trial.additivity > 1e-6:
            reason = (f"attribution of the trial does not close (residual "
                      f"{trial.additivity:.2e}); fix the game before trusting it")
            say(reason)
            break
        rounds.append(trial)
        dropped.append(candidate.name)
        current = trial
        say(f"round {len(dropped)}: dropped {candidate.name!r} "
            f"(share {float(first.shares.get(candidate.name, 0.0)):.4f}), "
            f"{len(trial.columns)} columns, selection MAE {trial.selection_mae:.6f}, "
            f"test MAE {trial.test_mae:.6f}")

    best_index = int(np.argmin([m.selection_mae for m in rounds]))
    # ties go to the simplest model: among the rounds that are within the accepted
    # tolerance of the best selection MAE, take the one with the fewest columns.
    # With tolerance 0 ("only free drops") that is the last round at the minimum.
    floor = float(min(m.selection_mae for m in rounds))
    eligible = [i for i, m in enumerate(rounds)
                if float(m.selection_mae) <= floor + float(tolerance) + 1e-12]
    best_index = eligible[-1] if eligible else best_index
    kept = rounds[best_index].columns
    return EliminationResult(rounds=rounds, dropped=dropped, kept=kept,
                             stop_reason=reason, best_index=best_index)


def effects_table(result: EliminationResult) -> pd.DataFrame:
    """``group`` -> delta MAE of **keeping** it, from the elimination path.

    ``delta_mae = MAE(round i) - MAE(round i+1)``, i.e. the negative of what the
    drop cost.  The sign is the one ``utils.xai.compare_with_ablation`` expects:
    negative means the group helped, positive means removing it was free.

    This replaces the L0-L5 ladder as the ablation source: instead of three fixed
    rungs measuring "does the block help when added", it measures "what does it
    cost when the model's own attributions say it can go".
    """
    rows = []
    for index, group in enumerate(result.dropped):
        if index >= len(result.rounds) - 1:
            break
        rows.append({
            "group": group,
            "delta_mae": float(result.rounds[index].selection_mae
                               - result.rounds[index + 1].selection_mae),
            "from_round": index,
            "to_round": index + 1,
        })
    return pd.DataFrame(rows, columns=["group", "delta_mae", "from_round", "to_round"])


def load_effects(path) -> pd.DataFrame:
    """Read ``elimination_rounds.csv`` back into an effects table."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["group", "delta_mae", "from_round", "to_round"])
    ledger = pd.read_csv(path)
    if not {"dropped_next", "selection_mae"} <= set(ledger.columns):
        return pd.DataFrame(columns=["group", "delta_mae", "from_round", "to_round"])
    rows = []
    for index in range(len(ledger) - 1):
        group = ledger.loc[index, "dropped_next"]
        if not isinstance(group, str) or not group.strip():
            continue
        rows.append({
            "group": group.strip(),
            "delta_mae": float(ledger.loc[index, "selection_mae"]
                               - ledger.loc[index + 1, "selection_mae"]),
            "from_round": index,
            "to_round": index + 1,
        })
    return pd.DataFrame(rows, columns=["group", "delta_mae", "from_round", "to_round"])


def write_outputs(result: EliminationResult, out_dir) -> Dict[str, str]:
    """Write the ledger, the per-round group shares and the kept column list."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rounds": out_dir / "elimination_rounds.csv",
        "shares": out_dir / "elimination_shares.csv",
        "kept": out_dir / "columns_kept.txt",
        "json": out_dir / "elimination.json",
        "effects": out_dir / "elimination_effects.csv",
    }
    result.ledger().to_csv(paths["rounds"], index=False, float_format="%.8f")
    effects_table(result).to_csv(out_dir / "elimination_effects.csv", index=False,
                                 float_format="%.8f")
    result.share_ledger().to_csv(paths["shares"], index=False, float_format="%.6f")
    paths["kept"].write_text("\n".join(result.kept) + "\n", encoding="utf-8")
    import json

    paths["json"].write_text(json.dumps({
        "kept": list(result.kept),
        "dropped": list(result.dropped),
        "stop_reason": result.stop_reason,
        "best_round": result.best_index,
        "selection_mae": result.best.selection_mae,
        "test_mae": result.best.test_mae,
        "baseline": {"selection_mae": result.rounds[0].selection_mae,
                     "test_mae": result.rounds[0].test_mae,
                     "n_columns": len(result.rounds[0].columns)},
    }, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}
