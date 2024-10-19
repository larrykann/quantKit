import unittest
import numpy as np
from pyQuantTools.math.math_helpers import LogReturns

class TestMathHelpers(unittest.TestCase):
    def setUp(self):
        # Example data for testing
        np.random.seed(42)  # For reproducibility
        self.values = np.random.rand(1000) + 1  # Avoid zeros for log returns
        self.small_values = np.random.rand(10) + 1  # Avoid zeros
        self.nan_values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

    def test_log_returns(self):
        # Test for proper output
        window = 1
        log_returns = LogReturns(self.values, window=window)
        self.assertIsInstance(log_returns, np.ndarray)
        self.assertEqual(len(log_returns), len(self.values))
        self.assertEqual(log_returns[:window].tolist(), [0] * window)

        # Test for a different window
        window = 5
        log_returns = LogReturns(self.values, window=window)
        self.assertEqual(len(log_returns), len(self.values))
        self.assertEqual(log_returns[:window].tolist(), [0] * window)

        # Test to check if NaN values raise an error
        with self.assertRaises(ValueError):
            LogReturns(self.nan_values)

if __name__ == "__main__":
    unittest.main()
