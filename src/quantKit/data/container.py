"""quantKit.data.container

Provides DataContainer for timestamp-agnostic time series storage backed by numpy arrays.
"""

__all__ = ["DataContainer"]

from typing import Dict, List, Any
import numpy as np

class DataContainer:
    """
    Container for time series data backed by numpy arrays.

    Args:
        timestamps (np.ndarray): 1D numpy array of datetime64 timestamps, sorted and unique.

    Attributes:
        timestamps (np.ndarray): Stored timestamps.
        _arrays (Dict[str, np.ndarray]): Mapping of field names to their numpy arrays.
    """
    def __init__(self, timestamps: np.ndarray):
        if timestamps.dtype.kind != 'M':
            raise TypeError("timestamps must be of dtype datetime64")
        if not np.all(np.diff(timestamps) >= np.timedelta64(0, 'ns')):
            raise ValueError("timestamps must be sorted non-decreasing")
        if len(np.unique(timestamps)) != len(timestamps):
            raise ValueError("timestamps must be unique")
        object.__setattr__(self, 'timestamps', timestamps)
        object.__setattr__(self, '_arrays', {})  # type: Dict[str, np.ndarray]

    def __getattr__(self, name: str) -> np.ndarray:
        """
        Retrieve an array field by name.

        Args:
            name (str): Field name to retrieve.

        Returns:
            np.ndarray: The array associated with the given field.

        Raises:
            AttributeError: If the field does not exist.
        """
        if name in self._arrays:
            return self._arrays[name]
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set a new field or overwrite an existing one.

        Args:
            name (str): Field name to set.
            value: Scalar or array-like value to store. Scalars are broadcast.

        Raises:
            ValueError: If array length does not match timestamps.
        """
        if name in ('timestamps', '_arrays'):
            object.__setattr__(self, name, value)
        else:
            arr = np.asarray(value)
            # scalar broadcast
            if arr.ndim == 0:
                arr = np.full(self.timestamps.shape[0], arr)
            if arr.shape[0] != self.timestamps.shape[0]:
                raise ValueError(
                    f"Length of '{name}' must match timestamps length {self.timestamps.shape[0]}"
                )
            self._arrays[name] = arr

    def to_numpy(self, fields: List[str]) -> np.ndarray:
        """
        Stack named fields into a 2D numpy array with shape (n_rows, len(fields)).

        Args:
            fields (List[str]): List of field names to include.

        Returns:
            np.ndarray: 2D array of shape (len(timestamps), len(fields)).

        Raises:
            ValueError: If any requested field is not present.
        """
        try:
            cols = [self._arrays[field] for field in fields]
        except KeyError as e:
            raise ValueError(f"Field {e.args[0]} not found in DataContainer")
        return np.column_stack(cols)
