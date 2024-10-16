import unittest
import numpy as np
from pyQuantTools.stats.stat_helpers import iqr, range_iqr_ratio, relative_entropy, simple_stats, fast_exponential_smoothing

class TestStatHelpers(unittest.TestCase):
    def setUp(self):
        # Example data for testing
        np.random.seed(42)  # For reproducibility
        self.values = np.random.rand(1000)
        self.small_values = np.random.rand(10)
        self.large_values = np.random.rand(20000)

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

if __name__ == "__main__":
    unittest.main()

