"""Physical weather and solar-geometry features (L2).

All functions are pure and accept plain sequences so they can be unit-tested
without touching disk.  Every time-based parameter is expressed in hours and
converted to steps by the caller: a 96-step window is 24 h at 15 min but 48 h at
30 min, and getting that wrong silently changes the feature.
"""

import math

import pandas as pd

DEFAULT_BASE_TEMPERATURE = 15.5
MAX_INTERPOLATION_HOURS = 3.0
SOLAR_CONSTANT_W_M2 = 1361.0

# ERA5 columns the feature store expects, in the order they are requested.
WEATHER_COLUMNS = (
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "wind_speed_10m",
    "precipitation",
)


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except TypeError:
        return False


def compute_degree_days(temperature_values, base_temperature=DEFAULT_BASE_TEMPERATURE):
    """Heating and cooling degree *hours* per step.

    The literature name is degree days; at sub-daily resolution the value is a
    degree-hour contribution, which is what the model consumes.
    """
    hdd_list, cdd_list = [], []
    for val in temperature_values:
        if _is_missing(val):
            hdd_list.append(None)
            cdd_list.append(None)
        else:
            hdd_list.append(max(base_temperature - val, 0.0))
            cdd_list.append(max(val - base_temperature, 0.0))
    return hdd_list, cdd_list


def compute_apparent_temperature(temp_celsius, relative_humidity, wind_speeds):
    """Steadman/Australian apparent temperature.

    AT = Ta + 0.33 * e - 0.70 * ws - 4.00, with e the water-vapour pressure in hPa.
    """
    apparent_temps = []
    for ta, rh, ws in zip(temp_celsius, relative_humidity, wind_speeds):
        if any(_is_missing(v) for v in (ta, rh, ws)):
            apparent_temps.append(None)
            continue
        vapor_pressure = (rh / 100.0) * 6.105 * math.exp((17.27 * ta) / (237.7 + ta))
        apparent_temps.append(ta + 0.33 * vapor_pressure - 0.70 * ws - 4.00)
    return apparent_temps


def calculate_solar_elevation(dt_sequence, latitude=51.5, longitude=0.0):
    """Solar elevation in degrees for UTC timestamps at (latitude, longitude)."""
    elevations = []
    lat_rad = math.radians(latitude)

    for item in dt_sequence:
        dt = item if hasattr(item, "hour") else pd.Timestamp(item).to_pydatetime()
        day_of_year = dt.timetuple().tm_yday

        declination_rad = math.radians(23.45) * math.sin(math.radians((360.0 / 365.0) * (day_of_year - 81)))

        b_angle = math.radians((360.0 / 365.0) * (day_of_year - 81))
        eot_minutes = 9.87 * math.sin(2.0 * b_angle) - 7.53 * math.cos(b_angle) - 1.5 * math.sin(b_angle)

        # Timestamps are UTC, so the solar-time offset is 4 minutes per degree of
        # longitude plus the equation of time.
        utc_hours = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        true_solar_time = (utc_hours + (4.0 * longitude + eot_minutes) / 60.0) % 24.0
        hour_angle_rad = math.radians(15.0 * (true_solar_time - 12.0))

        sin_elevation = (
            math.sin(lat_rad) * math.sin(declination_rad)
            + math.cos(lat_rad) * math.cos(declination_rad) * math.cos(hour_angle_rad)
        )
        elevations.append(math.degrees(math.asin(max(-1.0, min(1.0, sin_elevation)))))

    return elevations


def clear_sky_ghi(solar_elevation_deg):
    """Haurwitz clear-sky global horizontal irradiance (W/m^2).

    The PV task needs a radiation term that is known over the whole horizon:
    geometry-only clearsky cannot leak the measured PV output, while measured
    irradiance at future steps would.
    """
    out = []
    for elevation in solar_elevation_deg:
        if _is_missing(elevation) or elevation <= 0:
            out.append(0.0)
            continue
        cos_zenith = max(math.sin(math.radians(elevation)), 1e-6)
        out.append(SOLAR_CONSTANT_W_M2 * cos_zenith * math.exp(-0.057 / cos_zenith) * 0.75)
    return out


def compute_causal_rolling_stats(values_sequence, window_size):
    """Backward rolling mean/std over ``window_size`` steps (never looks ahead)."""
    rolling_means, rolling_stds = [], []

    for i in range(len(values_sequence)):
        start_idx = max(0, i - window_size + 1)
        window = [v for v in values_sequence[start_idx: i + 1] if not _is_missing(v)]
        if not window:
            rolling_means.append(None)
            rolling_stds.append(None)
            continue
        mean_val = sum(window) / len(window)
        rolling_means.append(mean_val)
        if len(window) > 1:
            var_val = sum((x - mean_val) ** 2 for x in window) / (len(window) - 1)
            rolling_stds.append(math.sqrt(var_val))
        else:
            rolling_stds.append(0.0)

    return rolling_means, rolling_stds


def interpolate_bounded_weather(data_sequence, max_gap_steps=12):
    """Linearly interpolate interior gaps of at most ``max_gap_steps`` steps.

    Longer gaps stay missing and gaps touching the ends are not extrapolated.
    """
    result = list(data_sequence)
    n = len(result)
    i = 0

    while i < n:
        if _is_missing(result[i]):
            gap_start = i
            while i < n and _is_missing(result[i]):
                i += 1
            gap_len = i - gap_start
            if gap_len <= max_gap_steps and gap_start > 0 and i < n:
                left, right = result[gap_start - 1], result[i]
                if not _is_missing(left) and not _is_missing(right):
                    step = (right - left) / (gap_len + 1)
                    for k in range(gap_len):
                        result[gap_start + k] = left + step * (k + 1)
        else:
            i += 1

    return result


def resample_to_grid(hourly, target_index, freq, max_gap_hours=MAX_INTERPOLATION_HOURS):
    """Resample hourly weather onto the series grid.

    Linear interpolation is allowed only for interior gaps of at most
    ``max_gap_hours``.  Anything longer, and every point outside the observed
    span, stays NaN and is reported by the caller.

    Returns ``(frame, report)`` with report mapping column -> fraction of the
    target grid that had to be interpolated.
    """
    if hourly is None or len(hourly) == 0:
        raise ValueError("resample_to_grid requires a non-empty hourly frame")

    minutes = int(pd.Timedelta(freq) / pd.Timedelta(minutes=1))
    max_gap_steps = int(max_gap_hours * (60 // minutes))

    target = pd.DatetimeIndex(target_index)
    hourly = hourly.copy()
    hourly.index = pd.DatetimeIndex(hourly.index)
    hourly = hourly[~hourly.index.duplicated(keep="first")].sort_index()

    aligned = hourly.reindex(hourly.index.union(target))
    interpolated = aligned.interpolate(method="time", limit=max_gap_steps, limit_area="inside")
    observed = aligned.notna().reindex(target).fillna(False)

    out = interpolated.reindex(target)
    report = {}
    for column in out.columns:
        filled = out[column].notna()
        filled_by_interp = int((filled & ~observed[column]).sum())
        report[column] = filled_by_interp / len(target) if len(target) else 0.0

    return out, report
