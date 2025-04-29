"""quantKit.data.validation

Provides functions to validate that a DataContainer’s fields and dtypes
conform to the official schemas defined in quantKit.data.schemas.

Primary entry point:
    validate_schema(dc: DataContainer, schema: dict) -> bool
"""

import numpy as np
from quantKit.data.container import DataContainer

def validate_schema(dc: DataContainer, schema: dict):
    """Validate a DataContainer against a given schema dictionary.

    Args:
        dc (DataContainer): The container whose fields and dtypes to check.
        schema (dict): Mapping from field names to expected NumPy dtypes.
            Must include the special key "timestamps" for the timestamp dtype.

    Returns:
        bool: Always True when validation passes.

    Raises:
        TypeError: If `timestamps` dtype or any field’s dtype doesn’t match.
        ValueError: If any required field is missing from the container.
    """
    ts = dc.timestamps
    expected_ts = schema["timestamps"]
    if str(ts.dtype) != str(expected_ts):
        raise TypeError(f"timestamps dtype {ts.dtype} ≠ expected {expected_ts!r}")

    # 2) For each schema entry, verify presence and dtype
    for field, expected_dtype in schema.items():
        if field == "timestamps":
            continue
        arr = dc._arrays.get(field)
        if arr is None:
            raise ValueError(f"Missing field '{field}' in DataContainer")
        if isinstance(expected_dtype, np.dtype):
            if arr.dtype != expected_dtype:
                raise TypeError(
                    f"Field '{field}' dtype {arr.dtype} ≠ expected {expected_dtype!r}"
                )
        else:
            # (if you ever support non-numpy-dtype schema entries)
            pass

    return True
