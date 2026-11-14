"""Iterate the five benchmark tasks through one loader interface.

Every model script needs the same series list: 4 PV + 4 BESS houses (Belgium),
the WPuQ household (Germany), the LCL household (London) and three Zonnedael
customers.  Keeping the list here means a new model cannot silently benchmark a
different set of series than the original table.
"""

from __future__ import annotations

import inspect
import logging

SERIES_ORDER = {
    "belgium": [f"pv_house_{house}" for house in (1, 2, 3, 4)]
    + [f"bess_house_{house}" for house in (1, 2, 3, 4)],
    "germany": ["germany_load"],
    "london": ["london_load"],
    "zonnedael": [f"zonnedael_customer_{customer}" for customer in (8, 9, 43)],
}

_DISPLAY = {
    "pv_house": "PV_house",
    "bess_house": "BESS_house",
    "germany_load": "germany_load",
    "london_load": "london_load",
    "zonnedael_customer": "zonnedael_customer",
}

# series prefix -> loader method on the dataset class
_LOADERS = {
    "pv_house": "get_inputs_for_pv",
    "bess_house": "get_inputs_for_battery",
    "germany_load": "get_inputs_for_load",
    "london_load": "get_inputs_for_load",
    "zonnedael_customer": "get_inputs_for_zonnedael_consumption",
}


def display_name(series_id: str) -> str:
    """Plot title used by the original scripts (``PV_house_1``, ``BESS_house_2``)."""
    prefix, _, suffix = series_id.rpartition("_")
    return f"{_DISPLAY.get(prefix, prefix)}_{suffix}" if prefix else series_id


def series_prefix(series_id: str) -> str:
    return series_id.rsplit("_", 1)[0]


def _loader_for(series_id: str) -> str:
    if series_id in ("germany_load", "london_load"):
        return _LOADERS[series_id]
    return _LOADERS[series_prefix(series_id)]


def _call(dataset, method_name, series_id, start, end, exog_level):
    method = getattr(dataset, method_name, None)
    if method is None:
        raise AttributeError(f"{type(dataset).__name__} has no loader {method_name!r}")

    accepted = inspect.signature(method).parameters
    prefix = series_prefix(series_id)
    candidates = {
        "exog_level": exog_level,
        "start_dt": start,
        "end_dt": end,
    }
    if prefix in ("pv_house", "bess_house"):
        candidates["house"] = int(series_id.rsplit("_", 1)[1])
    if prefix == "zonnedael_customer":
        candidates["customer_number"] = int(series_id.rsplit("_", 1)[1])

    # Only pass what the loader actually declares: London's NF loader takes no
    # window, Zonnedael's NF loader takes no window either.
    kwargs = {name: value for name, value in candidates.items()
              if name in accepted and value is not None}
    return method(**kwargs)


def iter_series(dataset_name, dataset, start=None, end=None, exog_level="L0"):
    """Yield ``(series_id, dataframe)`` for every task of ``dataset_name``.

    The dataframe is a long-format frame with ``unique_id``, ``ds``, ``y`` and,
    for ``exog_level != "L0"``, the exogenous columns of that level.  Pass the
    matching ``1D``/``NF`` dataset class: this order follows the NF loaders.
    """
    if dataset_name not in SERIES_ORDER:
        raise ValueError(f"Unknown dataset {dataset_name!r}; known: {sorted(SERIES_ORDER)}")
    for series_id in SERIES_ORDER[dataset_name]:
        method_name = _loader_for(series_id)
        frame = _call(dataset, method_name, series_id, start, end, exog_level)
        if len(frame) == 0:
            logging.warning("series %s came back empty", series_id)
        yield series_id, frame
