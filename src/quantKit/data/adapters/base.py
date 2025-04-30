"""quantKit.data.adapters.base

Generic base class for all dataset adapters, handling schema validation
and train/validation split by a fixed number of days.
"""

__all__ = ["BaseDatasetAdapter"]

from abc import ABC, abstractmethod
from quantKit.data.container import DataContainer
from quantKit.data.validation import validate_schema

class BaseDatasetAdapter(ABC):
    """
    Abstract base for dataset adapters.

    Fetches a full time series once, validates its schema, and splits it
    into training and validation sets based on the last N distinct days.

    Subclasses must implement the `_connect_and_prepare` method to produce
    a single `DataContainer` for the requested date range.
    """

    def __init__(
        self,
        *,
        schema: dict,
        start: str,
        end: str,
        validation_days: int,
    ):
        """
        Initialize the adapter by fetching and preparing data.

        Args:
            schema (dict): Mapping of field names to expected NumPy dtypes,
                used to validate the fetched data.
            start (str): Inclusive start date/time for the data request.
            end (str): Inclusive end date/time for the data request.
            validation_days (int): Number of most recent distinct days to reserve for validation.

        Raises:
            TypeError: If the fetched data do not conform to `schema`.
            ValueError: If there are fewer distinct days than `validation_days`.
        """
        # 1) Fetch the full dataset
        full_dc = self._connect_and_prepare((start, end))

        # 2) Validate against the schema
        validate_schema(full_dc, schema)

        # 3) Split into training and validation
        self.training_set, self.validation_set = full_dc.split_last_n_days(validation_days)

    @abstractmethod
    def _connect_and_prepare(self, date_range: tuple) -> DataContainer:
        """
        Fetch raw data for the given date range and return it as a DataContainer.

        Args:
            date_range (tuple): A 2-tuple (start, end) of date/time strings or objects.

        Returns:
            DataContainer: A container of timestamps and corresponding fields.
        """
        pass
