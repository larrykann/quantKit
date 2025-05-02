"""
quantKit.data

Core data handling facilities for financial time series.

This package provides comprehensive tools for working with financial data:

- **container**: Time series storage with numpy backend for efficient manipulation
- **schemas**: Standard data schemas for market data (trades, bars, daily data)
- **validation**: Schema validation utilities to ensure data integrity
- **generators**: Synthetic data generators for testing and development
- **adapters**: Data source connectors for various APIs, files, and data providers

The design philosophy of quantKit.data centers around the DataContainer class,
which provides a timestamp-indexed storage mechanism that maintains type safety
through schema validation while offering numpy-backed performance.

Typical workflow:

1. Load data through an adapter or generate synthetic data
2. Validate against standard schemas
3. Process and analyze using numpy operations
4. Split into training/validation sets for model development
"""

# Only expose the high-level modules, not individual classes
__all__ = [
    "container",
    "schemas",
    "validation", 
    "generators",
    "adapters"
]

# Import the submodules but don't import their contents
from . import container
from . import schemas
from . import validation
from . import generators
from . import adapters
