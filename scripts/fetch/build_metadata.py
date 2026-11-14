#!/usr/bin/env python3
"""Build the L4 static metadata tables from published values.

Belgium hardware numbers are transcribed from Ruddick et al., *Energy and AI* 18
(2024) 100448 -- Table A.6 (BESS), A.7 (inverters), A.8 (PV peak power, phase,
breaker).  The earlier code hard-coded different, unsourced capacities (5.0 kW /
10 kWh for every house), which silently mismatched the paper.

Nothing here is estimated: if a value is not in the paper it is not written.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.exogenous.config import METADATA_DIR  # noqa: E402

SOURCE = "Ruddick et al. 2024, Energy and AI 18:100448, Tables A.6-A.8"

BELGIUM_HARDWARE = [
    {
        "entity_id": "house_1",
        "pv_peak_kwp": 3.4,
        "bess_energy_kwh": 5.12,
        "bess_charge_kw": 3.2,
        "bess_discharge_kw": 3.2,
        "inverter_max_ac_kw": 3.0,
        "bess_model": "BYD HVS 5.1",
        "house_breaker_kva": 9.2,
        "ac_phase": "mono",
    },
    {
        "entity_id": "house_2",
        "pv_peak_kwp": 5.6,
        "bess_energy_kwh": 5.0,
        "bess_charge_kw": 2.5,
        "bess_discharge_kw": 2.5,
        "inverter_max_ac_kw": 5.0,
        "bess_model": "Huawei LUNA2000-5-EO",
        "house_breaker_kva": 17.2,
        "ac_phase": "three",
    },
    {
        "entity_id": "house_3",
        "pv_peak_kwp": 3.0,
        "bess_energy_kwh": 15.3,
        "bess_charge_kw": 3.0,
        "bess_discharge_kw": 4.0,
        "inverter_max_ac_kw": 6.0,
        "bess_model": "D-Centralized LF2",
        "house_breaker_kva": 9.2,
        "ac_phase": "mono",
    },
    {
        "entity_id": "house_4",
        "pv_peak_kwp": 2.6,
        "bess_energy_kwh": 3.55,
        "bess_charge_kw": 1.7,
        "bess_discharge_kw": 2.5,
        "inverter_max_ac_kw": 2.5,
        "bess_model": "Pylontech US2000B",
        "house_breaker_kva": 9.2,
        "ac_phase": "mono",
    },
]


def main():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    hardware = pd.DataFrame(BELGIUM_HARDWARE)
    hardware["source"] = SOURCE
    out = METADATA_DIR / "belgium_hardware.csv"
    hardware.to_csv(out, index=False)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}  ({len(hardware)} houses)")
    # No controller table is written: the EMS schedule exists only for the
    # release's Apr-Jun experiment, while the benchmark forecasts Jan-Mar, so
    # `controller_id` is n/a for Belgium by construction (data/exogenous/README.md
    # assumption 5).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
