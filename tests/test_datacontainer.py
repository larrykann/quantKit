import unittest
import numpy as np

from quantKit.data.container import DataContainer

class TestDataContainer(unittest.TestCase):
    def test_slice_by_mask(self):
        """slice_by_mask should keep only rows where mask is True."""
        ts = np.array([
            "2025-04-01T00:00:00.000000000",
            "2025-04-02T00:00:00.000000000",
            "2025-04-03T00:00:00.000000000",
            "2025-04-04T00:00:00.000000000",
        ], dtype="datetime64[ns]")
        dc = DataContainer(ts)
        # add a dummy field
        dc.value = np.array([10, 20, 30, 40], dtype=float)

        # mask: keep only 2nd and 4th entries
        mask = np.array([False, True, False, True])
        sliced = dc.slice_by_mask(mask)

        # timestamps should be the masked ones
        expected_ts = ts[mask]
        np.testing.assert_array_equal(sliced.timestamps, expected_ts)

        # and the field should be filtered too
        np.testing.assert_array_equal(sliced.value, np.array([20, 40], dtype=float))

    def test_split_last_n_days(self):
        """split_last_n_days should return correct train/validation splits."""
        # create 5 consecutive days
        days = np.array([
            "2025-04-01T00:00:00", "2025-04-02T00:00:00",
            "2025-04-03T00:00:00", "2025-04-04T00:00:00",
            "2025-04-05T00:00:00"
        ], dtype="datetime64[ns]")
        dc = DataContainer(days)
        dc.value = np.arange(5, dtype=float)  # [0,1,2,3,4]

        train, val = dc.split_last_n_days(2)
        # last 2 distinct days are 04-04 and 04-05
        np.testing.assert_array_equal(val.timestamps, days[-2:])
        np.testing.assert_array_equal(val.value, np.array([3.0, 4.0]))
        # training is everything before that
        np.testing.assert_array_equal(train.timestamps, days[:-2])
        np.testing.assert_array_equal(train.value, np.array([0., 1., 2.]))

    def test_split_last_n_days_not_enough(self):
        """Requesting more days than exist should raise ValueError."""
        days = np.array(["2025-04-01T00:00:00"], dtype="datetime64[ns]")
        dc = DataContainer(days)
        dc.value = np.array([1.0])
        with self.assertRaises(ValueError):
            dc.split_last_n_days(2)  # only 1 distinct day available

if __name__ == "__main__":
    unittest.main()
