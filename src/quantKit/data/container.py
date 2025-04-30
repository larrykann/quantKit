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
    
    def slice_by_mask(self, mask: np.ndarray) -> "DataContainer":
        """Return a new container filtered by a boolean mask.

        Args:
            mask (np.ndarray): 1D boolean array of the same length as `timestamps`;
                True to keep the row, False to drop it.

        Returns:
            DataContainer: New container containing only the masked rows.
        """
        filtered = DataContainer(self.timestamps[mask])
        for field, arr in self._arrays.items():
            filtered._arrays[field] = arr[mask]
        return filtered

    def split_last_n_days(self, n_days: int) -> tuple["DataContainer", "DataContainer"]:
        """Split into training and validation by the last N distinct calendar days.

        Treats each timestamp as a trading day via `datetime64[D]`, so weekends or
        holiday gaps are handled automatically.

        Args:
            n_days (int): Number of most recent distinct days to reserve for validation.

        Returns:
            Tuple[DataContainer, DataContainer]:
                - training_set: Container with all rows _before_ the validation window.
                - validation_set: Container with all rows on or _after_ the cutoff day.

        Raises:
            ValueError: If there are fewer than `n_days` distinct days in `timestamps`.
        """
        # Cast to dates to count unique days
        days = self.timestamps.astype("datetime64[D]")
        unique_days = np.unique(days)
        if unique_days.size < n_days:
            raise ValueError(
                f"Only {unique_days.size} distinct days available, "
                f"cannot split last {n_days} days."
            )
        cutoff = unique_days[-n_days]
        mask_val = days >= cutoff
        return self.slice_by_mask(~mask_val), self.slice_by_mask(mask_val)
