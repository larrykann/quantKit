"""quantKit.data.adapters.stock

Stock‐price data adapters.

This module defines:
- StockPriceDatasetAdapter: abstract interface for any stock‐price adapter.
- BaseStockAdapter:       concrete base class that fetches one series and splits it.
"""

__all__ = [
    "StockPriceDatasetAdapter",
    "BaseStockAdapter",
]

from abc import ABC, abstractmethod
from typing import Optional

from quantKit.data.adapters import BaseDatasetAdapter
from quantKit.data.schemas import DAILY_BAR_SCHEMA, INTRADAY_BAR_SCHEMA
from quantKit.data.container import DataContainer


class StockPriceDatasetAdapter(ABC):
    """Interface for adapters that provide stock‐price time series."""

    @property
    @abstractmethod
    def training_set(self) -> DataContainer:
        """DataContainer of in‐sample (training) data."""

    @property
    @abstractmethod
    def validation_set(self) -> DataContainer:
        """DataContainer of out‐of‐sample (validation) data."""


class BaseStockAdapter(BaseDatasetAdapter, StockPriceDatasetAdapter):
    """Base class for stock‐price adapters, handling schema + train/validation split."""

    def __init__(
        self,
        ticker: str,
        start:  str,
        end:    str,
        validation_days: int,
        intraday: bool = False,
    ):
        """Initialize and immediately fetch+split the OHLCV series.

        Args:
            ticker (str): Stock symbol to fetch.  Must be provided.
            start (str):  Inclusive start date, format "YYYY-MM-DD".
            end (str):    Inclusive end date, format "YYYY-MM-DD".
            validation_days (int): Number of most recent trading days to reserve for validation.
            intraday (bool): If True, use INTRADAY_BAR_SCHEMA; else DAILY_BAR_SCHEMA.

        Raises:
            ValueError: If `ticker` is None or an empty string.
        """
        if not ticker:
            raise ValueError("`ticker` must be provided to BaseStockAdapter")

        self.ticker: str = ticker
        """str: The ticker symbol this adapter is serving."""

        schema = INTRADAY_BAR_SCHEMA if intraday else DAILY_BAR_SCHEMA

        super().__init__(
            schema=schema,
            start=start,
            end=end,
            validation_days=validation_days,
        )

        # training_set & validation_set are now available as properties
