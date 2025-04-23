from typing import Dict, List
import numpy as np

class DataContainer:
    """
    Container for time series data backed by numpy arrays.
    Attributes can be set and accessed as named fields.
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
        if name in self._arrays:
            return self._arrays[name]
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name: str, value):
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
        """
        try:
            cols = [self._arrays[field] for field in fields]
        except KeyError as e:
            raise ValueError(f"Field {e.args[0]} not found in DataContainer")
        return np.column_stack(cols)
