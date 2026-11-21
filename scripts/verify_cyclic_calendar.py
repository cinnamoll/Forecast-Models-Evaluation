"""Check the cyclic calendar path end to end, without loading a checkpoint.

For each series it rebuilds exactly the frame ``scripts/xai_timeshap.py --cyclic``
builds, then answers three questions:

1. Is the counterfactual of the ``hour`` / ``dayofweek`` player really a shift of half
   a cycle (12 h / 3.5 days), or did the background fall back to a constant?
2. Does that counterfactual reach each model's input?  The three model inputs are
   built with ``utils.tsfm`` - no model, no weights.
3. Does the numeric channel survive routing?  For Timestral the two columns must land
   in ``known_covariates`` (numeric) and NOT in ``categorical_known_covariates``,
   which is the channel that target-encodes the clock into a climatology.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.xai_timeshap import load_long_frame, scaled_frame  # noqa: E402
from utils.covariates import (CYCLIC_CALENDAR, cyclic_calendar, cyclic_columns,  # noqa: E402
                              restrict_columns, roles_for)
from utils.protocol import horizon_for, steps_per_day  # noqa: E402
from utils.timeshap import apply_coalition, periodic_background, players_per_variable  # noqa: E402
from utils.tsfm import (chronos2_input, prepare_inputs, timesfm3_covariates,  # noqa: E402
                        timestral_arguments)

SAMPLING = 25.0
COLUMNS = ("hour", "dayofweek", "humidity", "temperature", "cloud_cover")
CASES = [("belgium", "bess_house_1"), ("belgium", "pv_house_1"), ("germany", "germany_load"),
         ("london", "london_load"), ("zonnedael", "zonnedael_customer_8")]


def delta(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.nanmax(np.abs(a - b))) if a.size else 0.0


rows = []
for dataset, series in CASES:
    horizon = horizon_for(SAMPLING)
    per_day = steps_per_day(SAMPLING, dataset)
    frame = load_long_frame(dataset, series, "COV", SAMPLING)
    frame, cycles = cyclic_calendar(frame, steps_per_day=per_day)
    keep = [name for column in COLUMNS
            for name in (cyclic_columns(column) if cyclic_columns(column)[0] in cycles else (column,))]
    scaled, roles = scaled_frame(frame, "COV", horizon)
    scaled, roles = restrict_columns(scaled, roles, keep)
    n_context = len(scaled) - horizon
    background = periodic_background(scaled, per_day, train_end=n_context, cycles=cycles)
    players = players_per_variable(scaled, roles, n_context)
    player = {p.name: p for p in players}

    on = apply_coalition(scaled, background, players, [p.name for p in players], mode="background")
    off = apply_coalition(scaled, background, players,
                          [p.name for p in players if p.name != "hour"], mode="background")

    payload_on = prepare_inputs(on, roles, "COV", horizon)
    payload_off = prepare_inputs(off, roles, "COV", horizon)
    tsl_on, tsl_off = timestral_arguments(payload_on), timestral_arguments(payload_off)
    chr_on, chr_off = chronos2_input(payload_on), chronos2_input(payload_off)
    tfm_on, tfm_off = timesfm3_covariates(payload_on), timesfm3_covariates(payload_off)

    def half_cycle_shift(base: str) -> float:
        """max |background - the same column half a cycle later| over the context."""
        worst = 0.0
        for name in cyclic_columns(base):
            period = cycles[name]
            values = scaled[name].to_numpy(dtype=float)
            phase = (np.arange(len(values)) % period + period // 2) % period
            worst = max(worst, float(np.max(np.abs(
                background[name].to_numpy(dtype=float)[:n_context] - values[phase][:n_context]))))
        return worst

    half_cycle = half_cycle_shift("hour")
    half_week = half_cycle_shift("dayofweek")

    def channels(arguments, key):
        block = arguments.get(key)
        return 0 if block is None else int(np.asarray(block).shape[1])

    rows.append({
        "run": f"{dataset}/{series}",
        "per_day": per_day,
        "hour_cols": len(player["hour"].columns),
        "hour_rows_cover_horizon": player["hour"].size == len(scaled),
        "dow_cols": len(player["dayofweek"].columns),
        "bg_vs_halfcycle": half_cycle,
        "timestral_known_numeric": channels(tsl_on, "known_covariates"),
        "timestral_known_categorical": channels(tsl_on, "categorical_known_covariates"),
        "dow_bg_vs_halfweek": half_week,
        "d_timestral": delta(tsl_on.get("known_covariates"), tsl_off.get("known_covariates")),
        "d_chronos2_past": delta(chr_on["past_covariates"]["hour_sin"], chr_off["past_covariates"]["hour_sin"]),
        "d_chronos2_future": delta(chr_on["future_covariates"]["hour_cos"], chr_off["future_covariates"]["hour_cos"]),
        "d_timesfm3": delta(tfm_on.get("past_future_covariates"), tfm_off.get("past_future_covariates")),
    })

t = pd.DataFrame(rows)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)
print(t.to_string(index=False, float_format=lambda v: f"{v:.3e}"))
print()
print("bg_vs_halfcycle : 0 means the 'hour off' counterfactual is exactly the clock 12 h")
print("                  later (the half-week shift for dayofweek is separate).")
print("d_*            : max |model input with hour on - with hour off|; 0 would mean the")
print("                  player is still a no-op, i.e. phi == 0 exactly.")
