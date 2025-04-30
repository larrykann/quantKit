from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Dict, Any, List
from quantKit.data.container import DataContainer
from .settings import StrategySettings, DEFAULT_STRATEGY_SETTINGS


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    def __init__(self):
        # merge defaults + overrides
        overrides = self.settings() or {}
        merged = {**asdict(DEFAULT_STRATEGY_SETTINGS), **overrides}
        self.settings = StrategySettings(**merged)

        # now validate required fields
        missing = [f for f in ("start_date","end_date","validation_days") 
                   if getattr(self.settings, f) is None]
        if missing:
            raise ValueError(
                f"Missing required settings: {', '.join(missing)}. "
                "Please override them in settings()."
            )

    @abstractmethod
    def settings(self) -> Dict[str, Any]:
       """
        Return only the fields you wish to override,
        e.g. {'start_date':'2021-01-01','end_date':'2025-04-23'}
        """
        return {}

    @abstractmethod
    def import_data(
            self,
            settings: StrategySettings
    ) -> Dict[str, DataContainer]:
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
