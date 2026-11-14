"""Calendar and holiday features (L0 / L1).

Two rules govern this module:

1. Timestamps are UTC.  Holiday dates and the DST flag are properties of local
   wall-clock time, so every calendar derivation first projects the index onto
   the dataset's local timezone.  Deriving them from raw UTC shifts UK/DE/NL
   holidays by one hour and mis-dates late-evening load.
2. Nothing here is approximated.  When a holiday source is missing the caller
   gets an explicit "unavailable" answer (``school_holiday_availability``), never
   a silent stand-in.
"""

import math
from datetime import date, datetime, timedelta

import pandas as pd

from utils.exogenous.config import HOLIDAY_DIR

SECONDS_IN_DAY = 86400.0
DEFAULT_COUNTRY_CODE = "GB"

# Fixed-date public holidays.  Sources: gov.uk (GB bank holidays),
# gesetze-im-internet.de + Niedersächsisches Feiertagsgesetz (DE-NI),
# rijksoverheid.nl (NL), belgium.be / FPS Finance (BE).  Movable days are
# computed from Easter below.
FIXED_HOLIDAYS = {
    "GB": {(1, 1), (12, 25), (12, 26)},
    "DE": {(1, 1), (5, 1), (10, 3), (12, 25), (12, 26)},
    "DE-NI": {(1, 1), (5, 1), (10, 3), (12, 25), (12, 26)},
    "NL": {(1, 1), (5, 5), (12, 25), (12, 26)},
    "BE": {(1, 1), (5, 1), (7, 21), (8, 15), (11, 1), (11, 11), (12, 25), (12, 26)},
}

# Good Friday / Easter Monday / Ascension / Whit Monday, as day offsets from
# Easter Sunday.  Good Friday is NOT a statutory public holiday in NL or BE, so
# it is absent from those tuples.
EASTER_OFFSETS = {
    "GB": (-2, 1),
    "GB-ENG": (-2, 1),
    "DE": (-2, 1, 39, 50),
    "DE-NI": (-2, 1, 39, 50),
    "NL": (1, 39, 50),
    "BE": (1, 39, 50),
}


def compute_easter_date(year_int):
    """Easter Sunday (Anonymous Gregorian algorithm)."""
    a = year_int % 19
    b = year_int // 100
    c = year_int % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year_int, month, day)


def _days_in_month(year, month):
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def _last_sunday(year, month):
    last = date(year, month, _days_in_month(year, month))
    return last.day - (last.weekday() - 6) % 7


def _nth_weekday(year, month, weekday, n):
    """n-th (1-based) weekday of a month; n == -1 means the last one."""
    if n == -1:
        last = date(year, month, _days_in_month(year, month))
        return last - timedelta(days=(last.weekday() - weekday) % 7)
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _uk_substitute(days):
    """UK rule: a fixed holiday falling on a weekend moves to the next free weekday."""
    bank = {d for d in days if d.weekday() < 5}
    for d in sorted(d for d in days if d.weekday() >= 5):
        candidate = d
        while True:
            candidate += timedelta(days=1)
            if candidate.weekday() < 5 and candidate not in bank:
                bank.add(candidate)
                break
    return bank


def public_holidays_for_year(year, country_code, subdivision=None):
    """Set of public-holiday dates for one country (or subdivision) and year."""
    key = country_code.upper()
    if key == "DE" and subdivision == "DE-NI":
        key = "DE-NI"

    days = {date(year, m, d) for m, d in FIXED_HOLIDAYS.get(key, set())}
    easter = compute_easter_date(year)
    for offset in EASTER_OFFSETS.get(key, ()):
        days.add(easter + timedelta(days=offset))

    if key in ("GB", "GB-ENG"):
        days.add(_nth_weekday(year, 5, 0, 1))    # Early May bank holiday
        days.add(_nth_weekday(year, 5, 0, -1))   # Spring bank holiday
        days.add(_nth_weekday(year, 8, 0, -1))   # Summer bank holiday
        days = _uk_substitute(days)

    if key == "DE-NI" and year >= 2018:
        days.add(date(year, 10, 31))             # Reformation Day, Lower Saxony

    if key == "NL":
        if year >= 2014:
            kings_day = date(year, 4, 27)        # King's Day from 2014
            days.add(kings_day - timedelta(days=1) if kings_day.weekday() == 6 else kings_day)
        else:
            queens_day = date(year, 4, 30)       # Queen's Day until 2013
            days.add(queens_day - timedelta(days=1) if queens_day.weekday() == 6 else queens_day)

    return days


def holiday_calendar(start_date, end_date, country_code, subdivision=None):
    """Union of public holidays covering a closed date range (with padding)."""
    first = int(pd.Timestamp(start_date).year)
    last = int(pd.Timestamp(end_date).year)
    out = set()
    for year in range(first - 1, last + 2):
        out |= public_holidays_for_year(year, country_code, subdivision)
    return out


def detect_regional_holidays(dt_sequence, country_code=DEFAULT_COUNTRY_CODE, subdivision=None):
    """Flag public holidays (1/0) for the local dates in ``dt_sequence``."""
    dates = [_as_date(item) for item in dt_sequence]
    if not dates:
        return []
    calendar = holiday_calendar(min(dates), max(dates), country_code, subdivision)
    return [1 if d in calendar else 0 for d in dates]


def detect_bridge_days(dt_sequence, country_code=DEFAULT_COUNTRY_CODE, subdivision=None):
    """Flag derived bridge days ("brugdag", "Brückentag").

    A bridge day is a Mon-Fri non-holiday squeezed between a public holiday and
    a weekend, or between two public holidays.  This is a documented heuristic
    derived from the holiday calendar, not a published list, and it is reported
    as derived in the paper.
    """
    dates = [_as_date(item) for item in dt_sequence]
    if not dates:
        return []
    calendar = holiday_calendar(min(dates), max(dates), country_code, subdivision)

    def weekend(d):
        return d.weekday() >= 5

    flags = []
    for d in dates:
        if weekend(d) or d in calendar:
            flags.append(0)
            continue
        prev_hol = (d - timedelta(days=1)) in calendar
        next_hol = (d + timedelta(days=1)) in calendar
        prev_weekend = weekend(d - timedelta(days=1))
        next_weekend = weekend(d + timedelta(days=1))
        flags.append(1 if (prev_hol and next_weekend) or (prev_weekend and next_hol) or (prev_hol and next_hol) else 0)
    return flags


def calculate_cyclic_features(dt_sequence):
    """Sine/cosine encodings for hour of day, day of week and month."""
    hour_sin, hour_cos = [], []
    dow_sin, dow_cos = [], []
    month_sin, month_cos = [], []

    for item in dt_sequence:
        dt = _as_datetime(item)
        hour_fraction = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
        dow_fraction = (dt.weekday() + hour_fraction) / 7.0
        month_fraction = ((dt.month - 1) + (dt.day - 1) / _days_in_month(dt.year, dt.month)) / 12.0

        hour_sin.append(math.sin(2.0 * math.pi * hour_fraction))
        hour_cos.append(math.cos(2.0 * math.pi * hour_fraction))
        dow_sin.append(math.sin(2.0 * math.pi * dow_fraction))
        dow_cos.append(math.cos(2.0 * math.pi * dow_fraction))
        month_sin.append(math.sin(2.0 * math.pi * month_fraction))
        month_cos.append(math.cos(2.0 * math.pi * month_fraction))

    return {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dayofweek_sin": dow_sin,
        "dayofweek_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }


def to_local_calendar(index, timezone_name, assume_utc=True):
    """Project a UTC index onto local wall-clock time (tz-naive).

    ``assume_utc`` mirrors how the benchmark loaders store time: the datasets
    ship UTC timestamps in tz-naive columns.  Pass ``False`` when the values are
    already local wall-clock time.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        return idx.tz_convert(timezone_name).tz_localize(None)
    if not assume_utc:
        return idx
    return idx.tz_localize("UTC").tz_convert(timezone_name).tz_localize(None)


def dst_flags_from_utc(index, timezone_name):
    """1 while local daylight saving time is in force, 0 otherwise."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(timezone_name)
    return [1 if ts.dst() != timedelta(0) else 0 for ts in local]


def check_daylight_saving_time(dt_sequence):
    """DST flag for timestamps that are already local time.

    Kept for callers that do not know their timezone; the timezone-correct path
    is :func:`dst_flags_from_utc`.
    """
    flags = []
    for item in dt_sequence:
        dt = _as_datetime(item)
        start = date(dt.year, 3, _last_sunday(dt.year, 3))
        end = date(dt.year, 10, _last_sunday(dt.year, 10))
        flags.append(1 if start <= dt.date() < end else 0)
    return flags


# --------------------------------------------------------------------------- #
# School holidays: used only from a published source, otherwise n/a.
# --------------------------------------------------------------------------- #

_SCHOOL_HOLIDAY_COLUMNS = ("country", "subdivision", "start", "end", "name")


def school_holiday_path(country_code):
    return HOLIDAY_DIR / f"school_holidays_{country_code.upper()}.csv"


def load_school_holidays(country_code):
    """Cached school-holiday table, or ``None`` when absent."""
    path = school_holiday_path(country_code)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    missing = [c for c in _SCHOOL_HOLIDAY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df


def school_holiday_availability(country_code, first_date, last_date, subdivision=None):
    """(available, reason).  Available only when the table covers the whole span."""
    df = load_school_holidays(country_code)
    if df is None:
        return False, f"no school-holiday table at {school_holiday_path(country_code)}"
    subset = df[df["country"].str.upper() == country_code.upper()]
    if subdivision is not None and "subdivision" in subset.columns:
        regional = subset[subset["subdivision"].isna() | (subset["subdivision"] == subdivision)]
        if not regional.empty:
            subset = regional
    if subset.empty:
        return False, f"no rows for {country_code}"
    start = pd.to_datetime(subset["start"])
    end = pd.to_datetime(subset["end"])
    wanted = set(range(pd.Timestamp(first_date).year, pd.Timestamp(last_date).year + 1))
    covered = set(range(start.min().year, end.max().year + 1))
    missing = sorted(wanted - covered)
    if missing:
        return False, f"school-holiday table does not cover {missing}"
    return True, "ok"


def school_holiday_flags(dates, country_code, subdivision=None):
    df = load_school_holidays(country_code)
    if df is None:
        return None
    subset = df[df["country"].str.upper() == country_code.upper()]
    if subdivision is not None:
        regional = subset[subset["subdivision"].isna() | (subset["subdivision"] == subdivision)]
        if not regional.empty:
            subset = regional
    spans = [
        (pd.Timestamp(s).date(), pd.Timestamp(e).date())
        for s, e in zip(subset["start"], subset["end"])
    ]
    return [1 if any(s <= d <= e for s, e in spans) else 0 for d in dates]


def build_l1_calendar_matrix(dt_sequence, country_code=DEFAULT_COUNTRY_CODE, subdivision=None, timezone_name=None):
    """L1 calendar block. Pass local wall-clock timestamps (see to_local_calendar)."""
    dates = [_as_date(item) for item in dt_sequence]

    features = calculate_cyclic_features(dt_sequence)
    features["is_holiday"] = detect_regional_holidays(dates, country_code, subdivision)
    features["is_bridge_day"] = detect_bridge_days(dates, country_code, subdivision)
    features["is_weekend"] = [1 if d.weekday() >= 5 else 0 for d in dates]

    if timezone_name is not None:
        features["is_dst"] = dst_flags_from_utc(pd.DatetimeIndex(dt_sequence), timezone_name)
    else:
        features["is_dst"] = check_daylight_saving_time(dates)

    # School holidays are deliberately *not* emitted: the tables cover 2024
    # (Belgium, WPuQ) but not 2013 (LCL, Zonnedael), and the ladder compares
    # datasets, so every dataset has to carry the same columns.  The loader and
    # its availability check stay below for the day all four windows are covered.
    return features


def _as_datetime(item):
    if isinstance(item, datetime):
        return item
    if isinstance(item, date):
        return datetime(item.year, item.month, item.day)
    return datetime.fromisoformat(str(item))


def _as_date(item):
    if isinstance(item, datetime):
        return item.date()
    if isinstance(item, date):
        return item
    return pd.Timestamp(item).date()
