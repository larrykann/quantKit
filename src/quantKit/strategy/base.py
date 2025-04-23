from abc import ABC, abstractmethod
from typing import Dict
from quantKit.data.container import DataContainer


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    @abstractmethod
    def settings(self) -> Dict:
        """
        Return strategy settings, which may include:
          - start_date (str 'YYYY-MM-DD')
          - end_date   (str)
          - timeframe  (str, e.g. '5m')
          - update_mode ('on_bar_close' or 'on_tick')
          - random_data (bool)
          - random_target_count (int)
          - random_params (dict)
          - results_dir (str)
        """
        pass

    @abstractmethod
    def import_data(self, settings: Dict) -> DataContainer:
        """
        Load data according to settings. If random_data=True,
        call generate_random_data, else load from CSV/API.
        Should handle merging multiple timeframes.
        """
        pass

    @abstractmethod
    def compute_features(self, data: DataContainer, settings: Dict) -> None:
        """
        Compute indicators and attach to `data` as attributes.
        """
        pass
