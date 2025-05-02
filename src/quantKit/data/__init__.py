"""
quantKit.data

Core data handling facilities for financial time series.

Primary components:
    
- container: Time series storage with numpy backend
- schemas: Standard data schemas for market data
- validation: Schema validation utilities
- generators: Synthetic data generators
- adapters: Data source connectors (APIs, files, etc.)
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
