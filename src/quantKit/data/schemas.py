"""quantKit.data.schemas

Defines core data schemas for DataContainer outputs.

These schemas establish standard field names and appropriate data types
for common financial datasets, enabling:

1. Consistent field naming across data sources
2. Automatic type validation
3. Interoperability between data adapters
4. Clear expectations for model inputs/outputs

Each schema maps field names to NumPy dtypes. Adapters must validate
DataContainer contents against these schemas.
"""

from dataclasses import dataclass, asdict
import numpy as np

__all__ = [
    "TRADE_SCHEMA",
    "INTRADAY_BAR_SCHEMA",
    "DAILY_BAR_SCHEMA",
]

@dataclass(frozen=True)
class TradeSchemaDef:
    """
    Schema for trade-level (tick) data.

    Attributes:
        timestamps (np.dtype): dtype of the timestamps array (datetime64[ns]).
        price (np.dtype): dtype of the price array (float64).
        volume (np.dtype): dtype of the volume array (float64).
    """
    timestamps: np.dtype = np.dtype("datetime64[ns]")
    price:      np.dtype = np.dtype("float64")
    volume:     np.dtype = np.dtype("float64")

# Runtime schema dict used by validate_schema()
TRADE_SCHEMA = asdict(TradeSchemaDef())

@dataclass(frozen=True)
class IntradayBarSchemaDef:
    """
    Schema for intraday bar data.

    Attributes:
        timestamps (np.dtype): datetime64[ns] array of bar boundaries.
        open (np.dtype): float64 open prices.
        high (np.dtype): float64 high prices.
        low (np.dtype): float64 low prices.
        close (np.dtype): float64 close prices.
        volume (np.dtype): float64 summed volumes.
    """
    timestamps: np.dtype = np.dtype("datetime64[ns]")
    open:       np.dtype = np.dtype("float64")
    high:       np.dtype = np.dtype("float64")
    low:        np.dtype = np.dtype("float64")
    close:      np.dtype = np.dtype("float64")
    volume:     np.dtype = np.dtype("float64")

INTRADAY_BAR_SCHEMA = asdict(IntradayBarSchemaDef())

@dataclass(frozen=True)
class DailyBarSchemaDef:
    """
    Schema for daily bar data with adjustments.

    Attributes:
        timestamps (np.dtype): datetime64[ns] array of calendar dates.
        open (np.dtype): float64 unadjusted open prices.
        high (np.dtype): float64 unadjusted high prices.
        low (np.dtype): float64 unadjusted low prices.
        close (np.dtype): float64 **adjusted** close prices.
        adj_close (np.dtype): float64 explicit adjusted close.
        volume (np.dtype): float64 unadjusted volumes.
        split_factor (np.dtype): float64 cumulative split factors.
    """
    timestamps:   np.dtype = np.dtype("datetime64[D]")
    open:         np.dtype = np.dtype("float64")
    high:         np.dtype = np.dtype("float64")
    low:          np.dtype = np.dtype("float64")
    close:        np.dtype = np.dtype("float64")
    adj_close:    np.dtype = np.dtype("float64")
    volume:       np.dtype = np.dtype("float64")
    split_factor: np.dtype = np.dtype("float64")

DAILY_BAR_SCHEMA = asdict(DailyBarSchemaDef())
