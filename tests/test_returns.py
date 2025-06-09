import unittest
import numpy as np

from quantKit.financial.returns import (
    simple_returns,
    log_returns,
    multi_period_simple_returns,
    multi_period_log_returns
)


class TestReturns(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.prices = np.array([100.0, 105.0, 110.0, 108.0, 112.0])
        # Calculate expected values precisely
        self.expected_simple = np.array([
            np.nan, 
            (105.0 - 100.0) / 100.0,  # 0.05
            (110.0 - 105.0) / 105.0,  # 0.047619047619...
            (108.0 - 110.0) / 110.0,  # -0.018181818...
            (112.0 - 108.0) / 108.0   # 0.037037037...
        ])
        self.expected_log = np.array([
            np.nan,
            np.log(105.0 / 100.0),
            np.log(110.0 / 105.0), 
            np.log(108.0 / 110.0),
            np.log(112.0 / 108.0)
        ])

    def test_simple_returns_basic(self):
        """Test basic simple returns calculation."""
        result = simple_returns(self.prices)
        
        # First element should be NaN
        self.assertTrue(np.isnan(result[0]))
        
        # Test remaining values with tolerance
        np.testing.assert_array_almost_equal(
            result[1:], self.expected_simple[1:], decimal=6
        )

    def test_simple_returns_multi_period(self):
        """Test simple returns with multiple periods."""
        result = simple_returns(self.prices, periods=2)
        
        # First two elements should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))
        
        # Test 2-period return: (110 - 100) / 100 = 0.1
        self.assertAlmostEqual(result[2], 0.1, places=6)
        
        # Test 2-period return: (108 - 105) / 105 ≈ 0.028571
        self.assertAlmostEqual(result[3], 0.028571, places=6)

    def test_simple_returns_validation(self):
        """Test input validation for simple returns."""
        # 2D array should raise error
        with self.assertRaises(ValueError):
            simple_returns(np.array([[100, 105], [110, 108]]))
        
        # Array too short should raise error
        with self.assertRaises(ValueError):
            simple_returns(np.array([100]))
        
        # Periods too large should raise error
        with self.assertRaises(ValueError):
            simple_returns(self.prices, periods=10)

    def test_log_returns_basic(self):
        """Test basic log returns calculation."""
        result = log_returns(self.prices)
        
        # First element should be NaN
        self.assertTrue(np.isnan(result[0]))
        
        # Test remaining values with tolerance
        np.testing.assert_array_almost_equal(
            result[1:], self.expected_log[1:], decimal=6
        )

    def test_log_returns_multi_period(self):
        """Test log returns with multiple periods."""
        result = log_returns(self.prices, periods=2)
        
        # First two elements should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))
        
        # Test 2-period log return: log(110/100) = log(1.1)
        expected = np.log(110.0/100.0)
        self.assertAlmostEqual(result[2], expected, places=6)

    def test_log_returns_validation(self):
        """Test input validation for log returns."""
        # Negative prices should raise error
        with self.assertRaises(ValueError):
            log_returns(np.array([100, -105, 110]))
        
        # Zero prices should raise error
        with self.assertRaises(ValueError):
            log_returns(np.array([100, 0, 110]))
        
        # 2D array should raise error
        with self.assertRaises(ValueError):
            log_returns(np.array([[100, 105], [110, 108]]))

    def test_multi_period_simple_returns(self):
        """Test multi-period simple returns calculation."""
        single_returns = simple_returns(self.prices)
        result = multi_period_simple_returns(single_returns, 3)
        
        # First three elements should be NaN
        self.assertTrue(np.all(np.isnan(result[:3])))
        
        # Test 3-period compound return
        # (1 + 0.05) * (1 + 0.047619) * (1 - 0.018182) - 1
        expected = (1.05 * 1.047619 * 0.981818) - 1
        self.assertAlmostEqual(result[3], expected, places=5)

    def test_multi_period_simple_returns_with_nan(self):
        """Test multi-period simple returns with NaN values."""
        returns_with_nan = np.array([np.nan, 0.05, np.nan, -0.02, 0.01])
        result = multi_period_simple_returns(returns_with_nan, 3)
        
        # Should handle NaN values properly
        self.assertTrue(np.all(np.isnan(result[:4])))
        self.assertTrue(np.isnan(result[2]))  # Window contains NaN

    def test_multi_period_simple_returns_validation(self):
        """Test input validation for multi-period simple returns."""
        returns = np.array([np.nan, 0.05, 0.03])
        
        # Array too short should raise error
        with self.assertRaises(ValueError):
            multi_period_simple_returns(returns, 5)
        
        # 2D array should raise error
        with self.assertRaises(ValueError):
            multi_period_simple_returns(np.array([[0.05, 0.03]]), 2)

    def test_multi_period_log_returns(self):
        """Test multi-period log returns calculation."""
        single_log_returns = log_returns(self.prices)
        result = multi_period_log_returns(single_log_returns, 3)
        
        # First three elements should be NaN
        self.assertTrue(np.all(np.isnan(result[:3])))
        
        # Test 3-period sum: sum of three consecutive log returns
        expected = (single_log_returns[1] + single_log_returns[2] + 
                   single_log_returns[3])
        self.assertAlmostEqual(result[3], expected, places=6)

    def test_multi_period_log_returns_with_nan(self):
        """Test multi-period log returns with NaN values."""
        log_returns_with_nan = np.array([np.nan, 0.049, np.nan, -0.018, 0.036])
        result = multi_period_log_returns(log_returns_with_nan, 2)
        
        # Should handle NaN values properly
        self.assertTrue(np.all(np.isnan(result[:2])))
        self.assertTrue(np.isnan(result[2]))  # Window contains NaN

    def test_multi_period_log_returns_validation(self):
        """Test input validation for multi-period log returns."""
        log_rets = np.array([np.nan, 0.049, 0.047])
        
        # Array too short should raise error
        with self.assertRaises(ValueError):
            multi_period_log_returns(log_rets, 5)
        
        # 2D array should raise error
        with self.assertRaises(ValueError):
            multi_period_log_returns(np.array([[0.049, 0.047]]), 2)

    def test_relationship_simple_and_log_returns(self):
        """Test relationship between simple and log returns."""
        simple_rets = simple_returns(self.prices)
        log_rets = log_returns(self.prices)
        
        # For small returns: log(1 + R) ≈ R
        # Skip NaN values
        valid_mask = ~np.isnan(simple_rets)
        
        # log(1 + simple_return) should equal log_return
        computed_log = np.log(1 + simple_rets[valid_mask])
        np.testing.assert_array_almost_equal(
            computed_log, log_rets[valid_mask], decimal=6
        )

    def test_edge_cases(self):
        """Test edge cases."""
        # Single price point (minimum valid case)
        prices_min = np.array([100.0, 105.0])
        simple_result = simple_returns(prices_min)
        log_result = log_returns(prices_min)
        
        self.assertTrue(np.isnan(simple_result[0]))
        self.assertTrue(np.isnan(log_result[0]))
        self.assertAlmostEqual(simple_result[1], 0.05, places=6)
        self.assertAlmostEqual(log_result[1], np.log(1.05), places=6)

    def test_large_price_movements(self):
        """Test with large price movements."""
        prices_volatile = np.array([100.0, 200.0, 50.0, 150.0])
        
        simple_rets = simple_returns(prices_volatile)
        log_rets = log_returns(prices_volatile)
        
        # Test 100% increase
        self.assertAlmostEqual(simple_rets[1], 1.0, places=6)
        self.assertAlmostEqual(log_rets[1], np.log(2.0), places=6)
        
        # Test 75% decrease
        self.assertAlmostEqual(simple_rets[2], -0.75, places=6)
        self.assertAlmostEqual(log_rets[2], np.log(0.25), places=6)

    def test_constant_prices(self):
        """Test with constant prices (zero returns)."""
        constant_prices = np.array([100.0, 100.0, 100.0, 100.0])
        
        simple_rets = simple_returns(constant_prices)
        log_rets = log_returns(constant_prices)
        
        # All returns except first should be zero
        self.assertTrue(np.isnan(simple_rets[0]))
        self.assertTrue(np.isnan(log_rets[0]))
        
        np.testing.assert_array_almost_equal(simple_rets[1:], 0.0, decimal=10)
        np.testing.assert_array_almost_equal(log_rets[1:], 0.0, decimal=10)


if __name__ == '__main__':
    unittest.main()
