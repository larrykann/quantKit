import unittest
import numpy as np
from datetime import datetime, timedelta
from pyQuantTools.reports.FeatureTestReport import run_indicator_tests

class TestRunIndicatorTests(unittest.TestCase):
    def setUp(self):
        # Generate sample features and target data
        np.random.seed(42)  # For reproducibility
        dates = np.array([(datetime.now().date() - timedelta(days=i)).strftime('%d/%m/%Y') for i in range(1000)][::-1])
        self.features = np.rec.array(
            [
                (dates[i], np.random.rand(), np.random.rand(), np.random.rand())
                for i in range(1000)
            ],
            dtype=[('Date', 'U10'), ('feature1', 'f8'), ('feature2', 'f8'), ('feature3', 'f8')]
        )
        self.target = np.rec.array(
            [
                (dates[i], np.random.rand())
                for i in range(1000)
            ],
            dtype=[('Date', 'U10'), ('target', 'f8')]
        )

    def test_run_indicator_tests_to_terminal(self):
        # Run the report function (check if it runs without errors)
        try:
            run_indicator_tests(
                features=self.features,
                target=self.target,
                report_name="Test_Report",
                save_plots_to_file=False,
                mi_params={'n_permutations': 1000},
                mcmbt_params={'n_permutations': 1000},
                threshold_params={'n_permutations': 1000}
            )
        except Exception as e:
            self.fail(f"run_indicator_tests raised an exception unexpectedly: {e}")

if __name__ == "__main__":
    # Run unit tests
    unittest.main()

