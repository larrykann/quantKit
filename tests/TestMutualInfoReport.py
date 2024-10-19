import unittest
import numpy as np
from pyQuantTools.reports.mutual_info_report import generate_mi_report

class TestMutualInformationReport(unittest.TestCase):
    def setUp(self):
        # Generate sample features and target data
        np.random.seed(42)  # For reproducibility
        self.features = np.rec.array(
            [
                (np.random.rand(), np.random.rand(), np.random.rand())
                for _ in range(1000)
            ],
            dtype=[('feature1', 'f8'), ('feature2', 'f8'), ('feature3', 'f8')]
        )
        self.target = np.rec.array(
            [
                (np.random.rand(),)
                for _ in range(1000)
            ],
            dtype=[('target', 'f8')]
        )

    def test_generate_mi_report(self):
        # Run the report function (check if it runs without errors)
        try:
            generate_mi_report(
                features=self.features,
                target=self.target,
                nbins_feature=10,
                nbins_target=10,
                n_permutations=10
            )
        except Exception as e:
            self.fail(f"generate_mi_report raised an exception unexpectedly: {e}")

if __name__ == "__main__":
    # Run unit tests
    unittest.main()

    # Generate and print the mutual information report
    features = np.rec.array(
        [
            (np.random.rand(), np.random.rand(), np.random.rand())
            for _ in range(1000)
        ],
        dtype=[('feature1', 'f8'), ('feature2', 'f8'), ('feature3', 'f8')]
    )
    target = np.rec.array(
        [
            (np.random.rand(),)
            for _ in range(1000)
        ],
        dtype=[('target', 'f8')]
    )
    generate_mi_report(features=features, target=target, nbins_feature=10, nbins_target=10, n_permutations=10)

