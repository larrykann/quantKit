import unittest
import pandas as pd
import numpy as np
from pyQuantTools.reports.threshold_report import generate_threshold_report

class TestGenerateThresholdReport(unittest.TestCase):
    def setUp(self):
        # Example data for testing
        np.random.seed(42)  # For reproducibility
        self.features = pd.DataFrame({
            'Indicator1': np.random.rand(100),
            'Indicator2': np.random.rand(100),
            'Date': pd.date_range(start='2020-01-01', periods=100)
        })

        self.target = pd.DataFrame({
            'Return': np.random.randn(100),
            'Date': pd.date_range(start='2020-01-01', periods=100)
        })

    def test_generate_threshold_report_valid(self):
        # Test with valid input
        try:
            generate_threshold_report(
                features=self.features,
                target=self.target,
                bins=13,
                min_cases_percent=5,
                n_permutations=100
            )
        except Exception as e:
            self.fail(f"generate_threshold_report raised an exception unexpectedly: {e}")

    def test_generate_threshold_report_invalid_bins(self):
        # Test with invalid bins parameter
        with self.assertRaises(ValueError):
            generate_threshold_report(
                features=self.features,
                target=self.target,
                bins=10  # Invalid value
            )

    def test_generate_threshold_report_invalid_features_type(self):
        # Test with invalid features type
        with self.assertRaises(ValueError):
            generate_threshold_report(
                features='invalid_type',
                target=self.target
            )

    def test_generate_threshold_report_invalid_target_type(self):
        # Test with invalid target type
        with self.assertRaises(ValueError):
            generate_threshold_report(
                features=self.features,
                target='invalid_type'
            )

    def test_generate_threshold_report_invalid_min_cases_percent(self):
        # Test with invalid min_cases_percent parameter
        with self.assertRaises(ValueError):
            generate_threshold_report(
                features=self.features,
                target=self.target,
                min_cases_percent=150  # Invalid value
            )

    def test_generate_threshold_report_negative_permutations(self):
        # Test with negative n_permutations
        with self.assertRaises(ValueError):
            generate_threshold_report(
                features=self.features,
                target=self.target,
                n_permutations=-1  # Invalid value
            )

if __name__ == "__main__":
    unittest.main()
