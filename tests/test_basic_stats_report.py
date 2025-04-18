import unittest
import numpy as np
import os
import tempfile

from quantKit.reports.basic_stats_report import generate_basic_stats_report

class TestBasicStatsReport(unittest.TestCase):
    def setUp(self):
        # Create a small recarray with two numeric fields
        dtype = [('Feature1', 'f8'), ('Feature2', 'f8')]
        self.data = np.array([
            (1.0, 10.0),
            (2.0, 20.0),
            (np.nan, 30.0),
            (4.0, np.nan),
            (5.0, 50.0),
        ], dtype=dtype).view(np.recarray)

    def test_no_save_runs(self):
        # Should run without error and print to console
        generate_basic_stats_report(self.data)

    def test_with_save_creates_csv(self):
        # Use a temp dir for CSV output
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_basic_stats_report(self.data, save_csv=True, csv_dir=tmpdir)
            files = os.listdir(tmpdir)
            # Exactly one CSV file
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].endswith('_basic_stats.csv'))
            # And it’s non-empty
            path = os.path.join(tmpdir, files[0])
            self.assertGreater(os.path.getsize(path), 0)
    
    
if __name__ == "__main__":
    unittest.main()
