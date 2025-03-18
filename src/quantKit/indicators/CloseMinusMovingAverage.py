import numpy as np
from pyQuantTools.stats.stat_helpers import atr, normal_cdf
from pyQuantTools.math.math_helpers import LogReturns
from pyQuantTools.indicators.Indicator import Indicator

class CMMA(Indicator):
    def __init__(self, data: dict, lookback: int, atr_lookback: int = 252):
        """
        CMMA Indicator using the Indicator abstract class.

        Parameters:
        - data (dict): A dictionary containing the input data required for the indicator calculation.
          Example: {'high': np.ndarray, 'low': np.ndarray, 'close': np.ndarray}
        - lookback (int): The lookback period for calculating the moving average.
        - atr_lookback (int): The lookback period for calculating the ATR.
        """
        super().__init__(data)
        self.lookback = lookback
        self.atr_lookback = atr_lookback

        # Type checks for input data
        high = data.get('high')
        low = data.get('low')
        close = data.get('close')
        
        if not isinstance(high, np.ndarray) or not isinstance(low, np.ndarray) or not isinstance(close, np.ndarray):
            raise TypeError("Input data must contain 'high', 'low', and 'close' as numpy arrays.")

        if high.dtype != np.float64 or low.dtype != np.float64 or close.dtype != np.float64:
            raise TypeError("Input arrays 'high', 'low', and 'close' must be of type np.float64.")

        self.data = {
            'high': high,
            'low': low,
            'close': close
        }

    def calculate(self) -> np.ndarray:
        # Extract data from the dictionary
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        # Ensure all input arrays have the same length and no NaN values
        if not (len(high) == len(low) == len(close)):
            raise ValueError("Input arrays 'high', 'low', and 'close' must have the same length.")
        if np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(close)):
            raise ValueError("Input arrays must not contain NaN values.")

        # Determine the greater lookback period for NaN initialization
        max_lookback = max(self.lookback, self.atr_lookback)

        # Initialize result array with NaN values for the first 'max_lookback' elements
        cmma_values = np.full(len(close), np.nan, dtype=np.float64)

        # Calculate CMMA values
        log_returns = LogReturns(close, window=1)
        atr_value = atr(high, low, close, period=self.atr_lookback, use_log=True)

        for i in range(max_lookback, len(close)):
            sum_log = np.sum(log_returns[i - self.lookback + 1:i + 1]) / self.lookback

            if atr_value[i] > 0.0:
                denom = atr_value[i] * np.sqrt(self.lookback + 1.0)
                cmma_raw = (log_returns[i] - sum_log) / denom
                cmma_values[i] = 100.0 * normal_cdf(1.0 * cmma_raw) - 50.0
            else:
                cmma_values[i] = 0.0

        self.result = cmma_values
        return cmma_values
