from pyQuantTools.indicators.Indicator import Indicator
from pyQuantTools.stats.stat_helpers import fast_exponential_smoothing
import numpy as np

class Stochastic(Indicator):
    def __init__(self, data: dict, period: int, smooth: int = 0):
        # Initialize the parent Indicator class
        super().__init__(data)
        self.period = period
        self.smooth = smooth

    def calculate(self) -> np.ndarray:
        closes = self.data['close']
        lows = self.data['low']
        highs = self.data['high']
        stoch_values = np.zeros(len(closes))

        for i in range(self.period - 1, len(closes)):
            high_period = np.max(highs[i - self.period + 1: i + 1])
            low_period = np.min(lows[i - self.period + 1: i + 1])
            stoch_values[i] = 100 * (closes[i] - low_period) / (high_period - low_period + 1e-10)

        if self.smooth == 1:
            stoch_values = fast_exponential_smoothing(stoch_values)
        elif self.smooth == 2:
            stoch_values = fast_exponential_smoothing(stoch_values)
            stoch_values = fast_exponential_smoothing(stoch_values)

        return stoch_values

