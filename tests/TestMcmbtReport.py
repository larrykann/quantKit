import unittest
import numpy as np
from pyQuantTools.reports.mcmbt_report import generate_mcmbt_report

class TestMCMBTReport(unittest.TestCase):
    def setUp(self):
        # Generate sample data for testing
        np.random.seed(42)  # For reproducibility
        self.data = np.rec.array(
            [
                (np.random.rand(), np.random.rand(), np.random.rand())
                for _ in range(1000)
            ],
            dtype=[('feature1', 'f8'), ('feature2', 'f8'), ('feature3', 'f8')]
        )

    def test_generate_mcmbt_report(self):
        # Run the report function (check if it runs without errors)
        try:
            generate_mcmbt_report(
                data=self.data,
                min_recent=100,
                max_recent=500,
                lag=1,
                n_permutations=10
            )
        except Exception as e:
            self.fail(f"generate_mcmbt_report raised an exception unexpectedly: {e}")

if __name__ == "__main__":
    # Run unit tests
    unittest.main()

    # Generate and print the MCMBT report
    data = np.rec.array(
        [
            (np.random.rand(), np.random.rand(), np.random.rand())
            for _ in range(1000)
        ],
        dtype=[('feature1', 'f8'), ('feature2', 'f8'), ('feature3', 'f8')]
    )
    generate_mcmbt_report(data=data, min_recent=100, max_recent=500, lag=1, n_permutations=10)

