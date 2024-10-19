import unittest
import numpy as np
from pyQuantTools.indicators.Indicator import Indicator
from pyQuantTools.indicators.Stochastic import Stochastic
from pyQuantTools.indicators.CloseMinusMovingAverage import CMMA

class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Generate a dictionary with 10,000 cases for open, high, low, close, and volume data
        np.random.seed(42)  # For reproducibility
        self.data = {
            'open': np.random.rand(10000),
            'high': np.random.rand(10000),
            'low': np.random.rand(10000),
            'close': np.random.rand(10000),
            'volume': np.random.randint(1, 1000, 10000)
        }

    def test_indicator_base_class(self):
        # Ensure that trying to instantiate Indicator directly raises an error
        with self.assertRaises(TypeError):
            indicator = Indicator(self.data)

    def test_stochastic_indicator(self):
        # Test Stochastic indicator with no smoothing, 1 layer, and 2 layers
        period = 14

        # Test with no smoothing
        stochastic_0 = Stochastic(data=self.data, period=period, smooth=0)
        stoch_values_0 = stochastic_0.calculate()
        self.assertEqual(len(stoch_values_0), len(self.data['close']))

        # Test with one level of smoothing
        stochastic_1 = Stochastic(data=self.data, period=period, smooth=1)
        stoch_values_1 = stochastic_1.calculate()
        self.assertEqual(len(stoch_values_1), len(self.data['close']))

        # Test with two levels of smoothing
        stochastic_2 = Stochastic(data=self.data, period=period, smooth=2)
        stoch_values_2 = stochastic_2.calculate()
        self.assertEqual(len(stoch_values_2), len(self.data['close']))

    def test_cmma_indicator(self):
        # Test CMMA indicator
        cmma_indicator = CMMA(data=self.data, lookback=14, atr_lookback=10)
        cmma_values = cmma_indicator.calculate()
        self.assertEqual(len(cmma_values), len(self.data['close']))
        self.assertTrue(np.isnan(cmma_values[:max(14, 10)]).all())  # Initial values should be NaN
        self.assertFalse(np.isnan(cmma_values[max(14, 10):]).any())  # Remaining values should not be NaN

if __name__ == "__main__":
    unittest.main()

