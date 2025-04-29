import unittest
import numpy as np

from quantKit.data.container import DataContainer
from quantKit.data.schemas import TRADE_SCHEMA, INTRADAY_BAR_SCHEMA, DAILY_BAR_SCHEMA
from quantKit.data.validation import validate_schema


class TestSchemasValidation(unittest.TestCase):
    def test_trade_schema_validates_correctly(self):
        ts = np.array([
            '2025-04-01T09:30:00',
            '2025-04-01T09:31:00'
        ], dtype='datetime64[ns]')
        dc = DataContainer(ts)
        dc.price = np.array([100.0, 101.5], dtype=np.float64)
        dc.volume = np.array([10.0, 15.0], dtype=np.float64)
        self.assertTrue(validate_schema(dc, TRADE_SCHEMA))

    def test_trade_schema_missing_field_raises(self):
        ts = np.array(['2025-04-01T09:30:00'], dtype='datetime64[ns]')
        dc = DataContainer(ts)
        dc.price = np.array([100.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            validate_schema(dc, TRADE_SCHEMA)

    def test_trade_schema_dtype_mismatch_raises(self):
        ts = np.array(['2025-04-01T09:30:00'], dtype='datetime64[ns]')
        dc = DataContainer(ts)
        dc.price = np.array([100.0], dtype=np.float64)
        dc.volume = np.array([10], dtype=np.int32)
        with self.assertRaises(TypeError):
            validate_schema(dc, TRADE_SCHEMA)

    def test_intraday_schema_validates_correctly(self):
        ts = np.array([
            '2025-04-01T09:30:00',
            '2025-04-01T09:31:00'
        ], dtype='datetime64[ns]')
        dc = DataContainer(ts)
        for field in ('open', 'high', 'low', 'close', 'volume'):
            setattr(dc, field, np.ones(len(ts), dtype=np.float64))
        self.assertTrue(validate_schema(dc, INTRADAY_BAR_SCHEMA))

    def test_daily_schema_validates_correctly(self):
        ts = np.array([
            '2025-04-01',
            '2025-04-02',
            '2025-04-03'
        ], dtype='datetime64[D]')
        dc = DataContainer(ts)
        for field in ('open', 'high', 'low', 'close', 'adj_close', 'volume', 'split_factor'):
            setattr(dc, field, np.full(len(ts), 1.0, dtype=np.float64))
        self.assertTrue(validate_schema(dc, DAILY_BAR_SCHEMA))

    def test_daily_schema_missing_fields_raises(self):
        ts = np.array(['2025-04-01', '2025-04-02'], dtype='datetime64[D]')
        dc = DataContainer(ts)
        dc.open = np.ones(len(ts), dtype=np.float64)
        dc.high = np.ones(len(ts), dtype=np.float64)
        with self.assertRaises(ValueError):
            validate_schema(dc, DAILY_BAR_SCHEMA)


if __name__ == '__main__':
    unittest.main()
