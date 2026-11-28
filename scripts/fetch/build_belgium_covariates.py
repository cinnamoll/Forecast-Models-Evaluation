#!/usr/bin/env python3
"""Build the Belgium covariates that the benchmark window can actually support.

``data/exogenous/metadata/Belgium/`` holds the file release of Ruddick et al.,
*Energy & AI* 18 (2024) 100448 (repository
``EVERGi/real_validation_saferl_treec_paper``).  The benchmark forecasts
2024-01-01 .. 2024-04-01, so only that window is read; the release's experiment
window (2024-04-08 .. 2024-06-17) is not used anywhere in this pipeline.

``day_ahead_prices.csv``
    The Belgian day-ahead quote the experiment was actually billed against
    (paper section 3.2.3), published per **kWh** on a 15-min grid.  Written as
    ``prices/day_ahead_BE_2024_paper.csv`` in the ENTSO-E cache schema
    (``time,price_eur_mwh``, hourly) so ``feature_store._build_price`` reads it
    through the same path as the API download and no model code changes.

``soc.csv``
    The BESS state of charge over the training window, written as
    ``metadata/belgium_state.csv`` (``entity_id,time,bess_soc``).  It is the one
    release column that is both a controller output and a legitimate covariate:
    it is the state the EMS starts from, so at a forecast origin it is measured,
    not predicted.  Kept as *past-only* (L6) and only for the BESS task.

Files in the release that are deliberately *not* turned into covariates:
``battery.csv``, ``grid.csv`` and ``energy_import/export.csv`` are what the EMS
under test did, so at a forecast origin they carry the answer rather than a
driver of it.  ``cons.csv`` is the load channel.  ``ems_used.csv`` and
``solar.csv`` only describe the experiment window, which is out of scope here.

Usage
-----
    python scripts/fetch/build_belgium_covariates.py            # build + report
    python scripts/fetch/build_belgium_covariates.py --check-only
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

#: The benchmark forecasts 2024-01-01 .. 2024-04-01 (data/README.md,
#: utils/protocol.py).  The release's experiment window (08-04 .. 17-06) is
#: deliberately not read: nothing in the benchmark uses it, and mixing a
#: different period into a covariate makes the column impossible to defend.

# The two windows the release ships, keyed by the token in the folder name.
PERIODS = {
    "1-1_1-4": ("2024-01-01 00:00:00", "2024-04-01 00:00:00"),  # training window
    "8-4_17-6": ("2024-04-08 15:00:00", "2024-06-17 15:00:00"),  # experiment window
}


def _duplicate_skips(duplicates):
    """Folders to ignore: every member of a duplicate group except the first.

    A byte-identical copy of another folder adds no data and, for a per-window
    output, would silently replace the real window with the wrong one.  The first
    folder of a group is the one whose name matched the expected window first.
    """
    skip = set()
    for group in duplicates:
        for _, _, path in group[1:]:
            skip.add(id(path))
    return skip


def _naive(series_or_value):
    """The release mixes ``...Z`` ISO stamps and naive stamps; both are UTC."""
    if isinstance(series_or_value, pd.Series):
        return pd.to_datetime(series_or_value.str.replace("Z", "", regex=False))
    return pd.Timestamp(str(series_or_value).replace("Z", "")).tz_localize(None)


def scan_release(base=None):
    """Map house number -> {period token: folder}, flagging duplicated folders."""
    base = Path(base or PROJECT_ROOT / "data" / "exogenous" / "Belgium")
    folder_re = re.compile(r"^house(\d+)_(1-1_1-4|8-4_17-6)(?:_environment)?$")
    if not base.is_dir():
        raise SystemExit(
            f"release directory not found: {base}\n"
            "  download the 'environment/' folders of "
            "EVERGi/real_validation_saferl_treec_paper into it first."
        )

    found = {}
    for path in sorted(base.iterdir()):
        if not path.is_dir():
            continue
        match = folder_re.match(path.name)
        if not match:
            continue
        house, period = int(match.group(1)), match.group(2)
        found.setdefault(house, {})[period] = path

    if not found:
        raise SystemExit(f"no house folder matched {folder_re.pattern} under {base}")

    # A folder that is a byte-identical copy of another one is a download
    # mistake, not a second period: it silently gives a house two copies of the
    # same window and no data for the other one.
    by_fingerprint = {}
    for house, periods in found.items():
        for period, path in periods.items():
            by_fingerprint.setdefault(folder_fingerprint(path), []).append((house, period, path))

    duplicates = [group for group in by_fingerprint.values() if len(group) > 1]
    return found, duplicates


def folder_fingerprint(path):
    """md5 over every CSV's name and content, so equal content collides."""
    digest = hashlib.md5()
    for csv in sorted(Path(path).glob("*.csv")):
        digest.update(csv.name.encode())
        digest.update(csv.read_bytes())
    return digest.hexdigest()


def _first_present(found, filename):
    """Path of ``filename`` in the first house folder that has it."""
    for house in sorted(found):
        for period in PERIODS:
            path = found[house].get(period)
            if path is not None and (Path(path) / filename).exists():
                return Path(path) / filename
    return None


def build_price_table(found, out_path=None):
    """Hourly EUR/MWh series from the release's per-kWh 15-min quote."""
    out_path = Path(out_path or PROJECT_ROOT / "data" / "exogenous" / "prices" / "day_ahead_BE_2024_paper.csv")
    source = _first_present(found, "day_ahead_prices.csv")
    if source is None:
        raise SystemExit("no day_ahead_prices.csv anywhere in the release")

    frame = pd.read_csv(source)
    price_column = frame.columns[1]
    frame = pd.DataFrame(
        {"time": _naive(frame["datetime"]), "price_eur_mwh": frame[price_column].astype(float) * 1000.0}
    ).sort_values("time")

    # The source repeats each hourly auction result on its four 15-min slots.
    # Keeping one row per hour is what every other price file here stores, and
    # the check below refuses to average a quarter that disagrees with its hour.
    hourly = frame.groupby(frame["time"].dt.floor("h")).agg(
        quote=("price_eur_mwh", "mean"),
        distinct=("price_eur_mwh", "nunique"),
        slots=("price_eur_mwh", "size"),
    )
    disagreeing = hourly[(hourly["distinct"] > 1) | (hourly["slots"] != 4)]
    if len(disagreeing):
        preview = ", ".join(str(ts) for ts in disagreeing.index[:3])
        raise SystemExit(
            f"{len(disagreeing)} hour(s) are not a single quote on four 15-min slots ({preview}); "
            "the day-ahead series is not the step function L3 assumes"
        )

    hourly.index.name = "time"
    table = hourly.reset_index()[["time", "quote"]].rename(columns={"quote": "price_eur_mwh"})
    table = table[["time", "price_eur_mwh"]].sort_values("time").reset_index(drop=True)

    expected_hours = 91 * 24  # 2024-01-01 .. 2024-03-31, leap year
    if len(table) != expected_hours:
        raise SystemExit(f"expected {expected_hours} hourly quotes, got {len(table)}")
    if table["time"].iloc[0] != pd.Timestamp("2024-01-01 00:00:00"):
        raise SystemExit(f"price series starts at {table['time'].iloc[0]}, after the window start")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False, float_format="%.5f")
    stats = {
        "rows": len(table),
        "first": table["time"].iloc[0],
        "last": table["time"].iloc[-1],
        "min": table["price_eur_mwh"].min(),
        "max": table["price_eur_mwh"].max(),
        "negative": int((table["price_eur_mwh"] < 0).sum()),
        "source": source,
    }
    return out_path, stats


def build_state_table(found, out_path=None, duplicates=()):
    """BESS state of charge per house and step, in [0, 1], long format.

    SOC is the one release column that is both a controller output and a
    legitimate covariate: it is the state the EMS starts from, so at a forecast
    origin it is measured, not predicted.  It is therefore kept as *past-only*
    (lagged by one horizon like weather), never as a known-future input, and it
    only describes the BESS task.

    Values outside [0, 1] are instrument faults, not states: house 1 records
    6.55 in the training window (a scale/serialisation slip).  They are dropped
    and counted rather than clipped, because clipping would invent a state.
    """
    skip = _duplicate_skips(duplicates)
    out_path = Path(out_path or PROJECT_ROOT / "data" / "exogenous" / "metadata" / "belgium_state.csv")
    # Only the 1-1_1-4 folders are read: the benchmark forecasts 2024-01-01 ..
    # 2024-04-01, so the experiment window (08-04 .. 17-06) is out of scope.
    used_period = "1-1_1-4"

    rows, faults = [], []
    for house in sorted(found):
        path = found[house].get(used_period)
        if path is None or id(path) in skip:
            continue
        source = Path(path) / "soc.csv"
        if not source.exists():
            faults.append(f"house_{house}: no soc.csv in the {used_period} folder")
            continue
        frame = pd.read_csv(source)
        column = frame.columns[1]
        frame = pd.DataFrame(
            {"entity_id": f"house_{house}", "time": _naive(frame["datetime"]), "bess_soc": frame[column].astype(float)}
        )
        out_of_range = int((frame["bess_soc"].notna() & ~frame["bess_soc"].between(0.0, 1.0)).sum())
        if out_of_range:
            faults.append(f"house_{house}/{used_period}: {out_of_range} rows outside [0,1] dropped")
        frame.loc[~frame["bess_soc"].between(0.0, 1.0), "bess_soc"] = float("nan")
        rows.append(frame)

    if not rows:
        raise SystemExit("no soc.csv anywhere in the release: the state block stays n/a")

    table = pd.concat(rows, ignore_index=True)
    table = table.drop_duplicates(subset=["entity_id", "time"], keep="last").sort_values(["entity_id", "time"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False, float_format="%.6f", date_format="%Y-%m-%d %H:%M:%S")
    coverage = table.groupby("entity_id")["bess_soc"].apply(lambda s: float(s.notna().mean()))
    return out_path, table, coverage.round(4).to_dict(), faults


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "exogenous" / "Belgium")
    parser.add_argument("--price-out", type=Path,
                        default=PROJECT_ROOT / "data" / "exogenous" / "prices" / "day_ahead_BE_2024_paper.csv")
    parser.add_argument("--state-out", type=Path,
                        default=PROJECT_ROOT / "data" / "exogenous" / "metadata" / "belgium_state.csv")
    parser.add_argument("--check-only", action="store_true", help="report without writing anything")
    args = parser.parse_args(argv)

    found, duplicates = scan_release(args.release_dir)

    print(f"release: {args.release_dir}")
    for house in sorted(found):
        for period, path in sorted(found[house].items()):
            window = " .. ".join(PERIODS[period])
            files = sorted(p.stem for p in Path(path).glob("*.csv"))
            print(f"  house_{house}  {period:10s} {window}  {len(files):2d} files")
    for group in duplicates:
        labels = ", ".join(f"house_{house}/{period}" for house, period, _ in group)
        print(f"  DUPLICATE content: {labels} -- only the first is used")

    absent = [
        f"house_{house}/{period}: {name}"
        for house in sorted(found)
        for period, path in sorted(found[house].items())
        for name in ("solar.csv", "battery.csv")
        if not (Path(path) / name).exists()
    ]
    if absent:
        print("  missing files in the release:")
        for line in absent:
            print(f"    - {line}")

    if args.check_only:
        print("\ncheck-only: nothing written")
        return 0

    price_path, stats = build_price_table(found, args.price_out)
    print(
        f"\nwrote {price_path.relative_to(PROJECT_ROOT)}  "
        f"{stats['rows']} hourly quotes {stats['first']} .. {stats['last']}, "
        f"{stats['min']:.2f} .. {stats['max']:.2f} EUR/MWh "
        f"({stats['negative']} negative)"
    )

    state_path, state, state_coverage, faults = build_state_table(found, args.state_out, duplicates)
    print(f"wrote {state_path.relative_to(PROJECT_ROOT)}  {len(state)} rows, "
          f"{state['entity_id'].nunique()} houses")
    for entity, fraction in state_coverage.items():
        print(f"  {entity}: {fraction:.1%} of steps carry a SOC")
    for fault in faults:
        print(f"  WARNING: {fault}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
