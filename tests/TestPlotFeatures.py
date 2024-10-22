import unittest
import numpy as np
from pyQuantTools.visualizations.PlotFeatures import plot_features
import sys

# Set encoding for stdout to handle special characters
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

class TestPlotFeaturesTerminal(unittest.TestCase):
    def setUp(self):
        # Load the CSV file as a recarray
        file_path = 'tests/data/@SPX/$SPX.csv'

        # Read the column names from the first line of the CSV file
        with open(file_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
        column_names = header_line.split(',')

        # Build the dtype list, setting 'Date' column to 'U8' (string of length 8), others to 'f8' (float)
        dtype_list = []
        for name in column_names:
            if name == 'Date':
                dtype_list.append(('Date', 'U10'))  # String of length 10 for 'yyyymmdd' dates
            else:
                dtype_list.append((name, 'f8'))  # 64-bit float for other columns

        # Now read the CSV file with specified dtype
        self.recarray = np.genfromtxt(
            file_path,
            delimiter=',',
            names=True,
            dtype=dtype_list,
            encoding='utf-8'
        )

        # Now the 'Date' column is strings in 'yyyymmdd' format
        dates = self.recarray['Date']

        # Convert dates from string 'yyyymmdd' to 'DD/MM/YYYY' format
        new_dates_formatted = []
        for date_str in dates:
            date_str = date_str.strip()  # Remove any leading/trailing whitespace
            # Ensure the date string is exactly 8 characters
            if len(date_str) != 8 or not date_str.isdigit():
                raise ValueError(f"Invalid date format: {date_str}")
            # Extract year, month, day
            year = date_str[0:4]
            month = date_str[4:6]
            day = date_str[6:8]
            new_date_str = f"{day}/{month}/{year}"
            new_dates_formatted.append(new_date_str)

        new_dates_formatted = np.array(new_dates_formatted, dtype='object')

        # Replace the 'Date' column in the recarray
        self.recarray['Date'] = new_dates_formatted

    def test_plot_features(self):
        print("Date column contents:", self.recarray['Date'])
        print("Date column dtype:", self.recarray['Date'].dtype)
        print("Type of first element in Date column:", type(self.recarray['Date'][0]))
        # Run the plot_features function (check if it runs without errors)
        try:
            plot_features(self.recarray)  # Plot the features
        except Exception as e:
            self.fail(f"plot_features raised an exception unexpectedly: {e}")

if __name__ == "__main__":
    # Run unit tests
    unittest.main()

