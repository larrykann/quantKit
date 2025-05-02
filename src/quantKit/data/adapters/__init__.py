# quantKit/data/adapters/__init__.py

"""
Adapter package for quantKit data sources.

This package provides base classes for data adapters:

- BaseDatasetAdapter: Generic one-pass fetch + split base class.
- StockPriceDatasetAdapter: Abstract interface for stock-price adapters.
- BaseStockAdapter: Concrete stock-specific adapter base.
"""

__all__ = [
    "BaseDatasetAdapter",
    "StockPriceDatasetAdapter",
    "BaseStockAdapter",
]

from .base import BaseDatasetAdapter
from .stock import StockPriceDatasetAdapter, BaseStockAdapter
