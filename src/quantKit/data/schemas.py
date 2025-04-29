"""quantKit.data.schemas

Defines core data schemas for DataContainer outputs.

Each schema maps field names to NumPy dtypes. Adapters must validate
DataContainer contents against these schemas.
"""

import numpy as np

__all__ = [
    "TRADE_SCHEMA",
    "INTRADAY_BAR_SCHEMA",
    "DAILY_BAR_SCHEMA",
]

"""
Trade-level (tick/trade) data schema.

Fields:
    timestamps (datetime64[ns]): array of trade timestamps.
    price (float64): array of trade prices.
    volume (float64): array of trade volumes.
"""
TRADE_SCHEMA = {
    "timestamps": np.dtype("datetime64[ns]"),
    "price":      np.dtype("float64"),
    "volume":     np.dtype("float64"),
}

"""
Intraday bar data schema.

Fields:
    timestamps (datetime64[ns]): array of bar boundary timestamps.
    open (float64): array of bar open prices.
    high (float64): array of bar high prices.
    low (float64): array of bar low prices.
    close (float64): array of bar close prices.
    volume (float64): array of bar volumes (sum of trade volumes).
"""
INTRADAY_BAR_SCHEMA = {
    "timestamps": np.dtype("datetime64[ns]"),
    "open":       np.dtype("float64"),
    "high":       np.dtype("float64"),
    "low":        np.dtype("float64"),
    "close":      np.dtype("float64"),
    "volume":     np.dtype("float64"),
}

"""
Daily bar data schema with adjustments.

Fields:
    timestamps (datetime64[D]): array of calendar dates.
    open (float64): array of unadjusted open prices.
    high (float64): array of unadjusted high prices.
    low (float64): array of unadjusted low prices.
    close (float64): array of adjusted close prices.
    adj_close (float64): explicit adjusted close, identical to close.
    volume (float64): array of unadjusted volumes.
    split_factor (float64): array of cumulative split factors.
"""
DAILY_BAR_SCHEMA = {
    "timestamps":   np.dtype("datetime64[D]"),
    "open":         np.dtype("float64"),
    "high":         np.dtype("float64"),
    "low":          np.dtype("float64"),
    "close":        np.dtype("float64"),
    "adj_close":    np.dtype("float64"),
    "volume":       np.dtype("float64"),
    "split_factor": np.dtype("float64"),
}

