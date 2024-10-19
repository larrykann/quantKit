import unittest
import numpy as np
from pyQuantTools.stats.stat_helpers import iqr, range_iqr_ratio, relative_entropy, simple_stats, fast_exponential_smoothing, atr, normal_cdf

class TestStatHelpers(unittest.TestCase):
    def setUp(self):
        # Example data for testing
        np.random.seed(42)  # For reproducibility
        self.values = np.random.rand(1000)
        self.small_values = np.random.rand(10)
        self.large_values = np.random.rand(20000)

        # For ATR function
        self.high_prices = np.random.rand(1000) * 100 + 100  # Random high prices between 100 and 200
        self.low_prices = self.high_prices - np.random.rand(1000) * 10  # Low prices, 0-10 less than high prices
        self.close_prices = self.low_prices + np.random.rand(1000) * 5  # Close prices, 0-5 greater than low prices

    def test_iqr(self):
        result = iqr(self.values)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_range_iqr_ratio(self):
        result = range_iqr_ratio(self.values)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_relative_entropy(self):
        result = relative_entropy(self.values)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_simple_stats(self):
        ncases, mean, min_value, max_value = simple_stats(self.values)
        self.assertEqual(ncases, len(self.values))
        self.assertIsInstance(mean, float)
        self.assertIsInstance(min_value, float)
        self.assertIsInstance(max_value, float)
        self.assertLessEqual(min_value, max_value)

    def test_relative_entropy_small_values(self):
        result = relative_entropy(self.small_values)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_relative_entropy_large_values(self):
        result = relative_entropy(self.large_values)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_fast_exponential_smoothing(self):
        alpha = 0.5
        smoothed_values = fast_exponential_smoothing(self.values, alpha=alpha)
        self.assertIsInstance(smoothed_values, np.ndarray)
        self.assertEqual(len(smoothed_values), len(self.values))
        # Check that the first value of the smoothed array is the same as the original
        self.assertEqual(smoothed_values[0], self.values[0])

    def test_atr(self):
        period = 14
        use_log = True
        atr_values = atr(self.high_prices, self.low_prices, self.close_prices, period=period, use_log=use_log)
        self.assertIsInstance(atr_values, np.ndarray)
        self.assertEqual(len(atr_values), len(self.high_prices))  # The ATR length should be the same as the input length
        self.assertTrue(np.isnan(atr_values[:period]).all())  # The first `period` values should be NaN
        self.assertFalse(np.isnan(atr_values[period:]).any())  # The rest of the values should not be NaN

    def test_normal_cdf(self):
        z_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected_results = np.array([0.0228, 0.1587, 0.5, 0.8413, 0.9772])
        for z, expected in zip(z_values, expected_results):
            result = normal_cdf(z)
            self.assertIsInstance(result, float)
            self.assertAlmostEqual(result, expected, places=2)

if __name__ == "__main__":
    unittest.main()

