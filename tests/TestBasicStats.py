import unittest
import numpy as np
import pandas as pd
from pyQuantTools.reports.basic_stats_report import generate_basic_stats_report

class TestBasicStatsReport(unittest.TestCase):
    def setUp(self):
        # Example data for testing
        np.random.seed(42)
        self.data = pd.DataFrame({
            'Feature1': np.random.rand(100),
            'Feature2': np.random.rand(100)
        })

    def test_generate_basic_stats_report(self):
        # Test if the report function runs without errors
        try:
            generate_basic_stats_report(self.data)
        except Exception as e:
            self.fail(f"generate_basic_stats_report raised an exception unexpectedly: {e}")

if __name__ == "__main__":
    unittest.main()
