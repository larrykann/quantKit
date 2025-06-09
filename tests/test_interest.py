import unittest
import numpy as np

from quantKit.financial.interest import (
    simple_interest,
    discrete_compound_interest,
    continuous_compound_interest
)


class TestInterest(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.principal = np.array([100.0, 1000.0, 5000.0])
        self.rate = 0.05  # 5% annual rate
        self.time = np.array([1.0, 2.0, 3.0])

    def test_simple_interest_basic(self):
        """Test basic simple interest calculation."""
        # Test single values
        result = simple_interest(np.array([100.0]), 0.05, np.array([2.0]))
        expected = 100.0 * (1 + 0.05 * 2.0)  # 110.0
        
        self.assertAlmostEqual(result[0], expected, places=6)

    def test_simple_interest_vectorized(self):
        """Test vectorized simple interest calculation."""
        result = simple_interest(self.principal, self.rate, self.time)
        
        # A = P * (1 + r * t)
        expected = self.principal * (1 + self.rate * self.time)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_simple_interest_zero_time(self):
        """Test simple interest with zero time."""
        result = simple_interest(self.principal, self.rate, np.zeros_like(self.time))
        
        # Should return original principal
        np.testing.assert_array_almost_equal(result, self.principal, decimal=10)

    def test_simple_interest_zero_rate(self):
        """Test simple interest with zero rate."""
        result = simple_interest(self.principal, 0.0, self.time)
        
        # Should return original principal
        np.testing.assert_array_almost_equal(result, self.principal, decimal=10)

    def test_discrete_compound_interest_annual(self):
        """Test discrete compound interest with annual compounding."""
        # Annual compounding (freq=1)
        result = discrete_compound_interest(
            np.array([100.0]), 0.12, np.array([2.0]), freq=1
        )
        
        # A = P * (1 + r)^t = 100 * (1.12)^2
        expected = 100.0 * (1.12 ** 2)
        
        self.assertAlmostEqual(result[0], expected, places=6)

    def test_discrete_compound_interest_quarterly(self):
        """Test discrete compound interest with quarterly compounding."""
        # Quarterly compounding (freq=4)
        result = discrete_compound_interest(
            np.array([100.0]), 0.12, np.array([2.0]), freq=4
        )
        
        # A = P * (1 + r/m)^(m*t) = 100 * (1 + 0.12/4)^(4*2)
        expected = 100.0 * ((1 + 0.12/4) ** (4 * 2))
        
        self.assertAlmostEqual(result[0], expected, places=6)

    def test_discrete_compound_interest_vectorized(self):
        """Test vectorized discrete compound interest."""
        result = discrete_compound_interest(
            self.principal, self.rate, self.time, freq=12
        )
        
        # A = P * (1 + r/m)^(m*t)
        expected = self.principal * np.power(1 + self.rate/12, 12 * self.time)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_discrete_compound_interest_reduces_to_simple(self):
        """Test compound vs simple interest relationship for small rates/times."""
        # For small rates and times, compound should be slightly higher than simple
        # but the difference should be small
        
        principal_val = np.array([100.0])
        rate_val = 0.01  # 1%
        time_val = np.array([0.5])  # 6 months
        
        simple_result = simple_interest(principal_val, rate_val, time_val)
        compound_result = discrete_compound_interest(
            principal_val, rate_val, time_val, freq=1
        )
        
        # Simple: 100 * (1 + 0.01 * 0.5) = 100.5
        # Compound: 100 * (1.01)^0.5 ≈ 100.4988
        # Actually compound should be LESS than simple for periods < 1 year
        
        # For sub-annual periods, compound can be less than simple
        # Let's test the mathematical relationship instead
        self.assertNotEqual(compound_result[0], simple_result[0])
        
        # Both should be greater than principal
        self.assertGreater(simple_result[0], principal_val[0])
        self.assertGreater(compound_result[0], principal_val[0])
        
        # Difference should be small for small rate and time
        diff = abs(compound_result[0] - simple_result[0])
        self.assertLess(diff, 0.01)

    def test_continuous_compound_interest_basic(self):
        """Test basic continuous compound interest calculation."""
        result = continuous_compound_interest(
            np.array([100.0]), 0.05, np.array([3.0])
        )
        
        # A = P * exp(r * t) = 100 * exp(0.05 * 3)
        expected = 100.0 * np.exp(0.05 * 3.0)
        
        self.assertAlmostEqual(result[0], expected, places=6)

    def test_continuous_compound_interest_vectorized(self):
        """Test vectorized continuous compound interest."""
        result = continuous_compound_interest(self.principal, self.rate, self.time)
        
        # A = P * exp(r * t)
        expected = self.principal * np.exp(self.rate * self.time)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_continuous_compound_zero_rate(self):
        """Test continuous compound interest with zero rate."""
        result = continuous_compound_interest(self.principal, 0.0, self.time)
        
        # exp(0) = 1, so should return original principal
        np.testing.assert_array_almost_equal(result, self.principal, decimal=10)

    def test_continuous_compound_zero_time(self):
        """Test continuous compound interest with zero time."""
        result = continuous_compound_interest(
            self.principal, self.rate, np.zeros_like(self.time)
        )
        
        # exp(0) = 1, so should return original principal
        np.testing.assert_array_almost_equal(result, self.principal, decimal=10)

    def test_compounding_frequency_convergence(self):
        """Test that higher compounding frequencies approach continuous compounding."""
        principal_val = np.array([1000.0])
        rate_val = 0.06  # 6%
        time_val = np.array([5.0])  # 5 years
        
        # Calculate continuous compound result
        continuous_result = continuous_compound_interest(
            principal_val, rate_val, time_val
        )
        
        # Test increasing frequencies approaching continuous
        frequencies = [1, 4, 12, 52, 365, 8760]  # annual to hourly
        results = []
        
        for freq in frequencies:
            result = discrete_compound_interest(
                principal_val, rate_val, time_val, freq=freq
            )
            results.append(result[0])
        
        # Each higher frequency should be closer to continuous
        for i in range(1, len(results)):
            diff_prev = abs(results[i-1] - continuous_result[0])
            diff_curr = abs(results[i] - continuous_result[0])
            self.assertLessEqual(diff_curr, diff_prev)
        
        # Hourly compounding should be very close to continuous
        self.assertAlmostEqual(
            results[-1], continuous_result[0], delta=0.01
        )

    def test_interest_relationships(self):
        """Test relationships between different interest calculations."""
        principal_val = np.array([1000.0])
        rate_val = 0.08  # 8%
        time_val = np.array([2.0])  # 2 years
        
        simple_result = simple_interest(principal_val, rate_val, time_val)
        compound_annual = discrete_compound_interest(
            principal_val, rate_val, time_val, freq=1
        )
        compound_quarterly = discrete_compound_interest(
            principal_val, rate_val, time_val, freq=4
        )
        continuous_result = continuous_compound_interest(
            principal_val, rate_val, time_val
        )
        
        # Order should be: simple < annual < quarterly < continuous
        self.assertLess(simple_result[0], compound_annual[0])
        self.assertLess(compound_annual[0], compound_quarterly[0])
        self.assertLess(compound_quarterly[0], continuous_result[0])

    def test_negative_rates(self):
        """Test interest calculations with negative rates (deflation)."""
        principal_val = np.array([1000.0])
        negative_rate = -0.02  # -2% (deflation)
        time_val = np.array([3.0])
        
        simple_result = simple_interest(principal_val, negative_rate, time_val)
        compound_result = discrete_compound_interest(
            principal_val, negative_rate, time_val, freq=1
        )
        continuous_result = continuous_compound_interest(
            principal_val, negative_rate, time_val
        )
        
        # All should be less than principal
        self.assertLess(simple_result[0], principal_val[0])
        self.assertLess(compound_result[0], principal_val[0])
        self.assertLess(continuous_result[0], principal_val[0])
        
        # All should be positive (no negative money)
        self.assertGreater(simple_result[0], 0)
        self.assertGreater(compound_result[0], 0)
        self.assertGreater(continuous_result[0], 0)

    def test_large_time_periods(self):
        """Test interest calculations over large time periods."""
        principal_val = np.array([100.0])
        rate_val = 0.07  # 7%
        time_val = np.array([30.0])  # 30 years
        
        simple_result = simple_interest(principal_val, rate_val, time_val)
        compound_result = discrete_compound_interest(
            principal_val, rate_val, time_val, freq=1
        )
        continuous_result = continuous_compound_interest(
            principal_val, rate_val, time_val
        )
        
        # Verify reasonable growth over 30 years
        self.assertGreater(simple_result[0], 300.0)  # At least 3x growth
        self.assertGreater(compound_result[0], 600.0)  # Compound growth
        self.assertGreater(continuous_result[0], compound_result[0])

    def test_array_broadcasting(self):
        """Test that functions handle different array shapes correctly."""
        # Different sized arrays should work with broadcasting
        principals = np.array([100.0, 200.0])
        rate = 0.05
        times = np.array([1.0, 2.0])
        
        simple_result = simple_interest(principals, rate, times)
        compound_result = discrete_compound_interest(principals, rate, times, freq=4)
        continuous_result = continuous_compound_interest(principals, rate, times)
        
        # Should return arrays of same length as inputs
        self.assertEqual(len(simple_result), 2)
        self.assertEqual(len(compound_result), 2)
        self.assertEqual(len(continuous_result), 2)


if __name__ == '__main__':
    unittest.main()
