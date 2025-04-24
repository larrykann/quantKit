from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class StrategySettings:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    intraday_start_date: Optional[str] = None

    def __post_init__(self):
        # intraday defaults to start_date if not specified
        if self.intraday_start_date is None and self.start_date is not None:
            self.intraday_start_date = self.start_date


