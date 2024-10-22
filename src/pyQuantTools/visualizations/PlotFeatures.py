import plotext as plx
import numpy as np
from typing import Optional

def plot_features(
    features: np.recarray,
    logo_path: Optional[str] = None
) -> None:
    """
    Plots all indicators in the given features over time and displays them in the terminal.

    Parameters:
    - features: recarray containing the data to be plotted.
    - logo_path (Optional[str]): Path to the logo image to be added to the plot (default is None).
    """
    columns_to_plot = [column for column in features.dtype.names if column != 'Date']
    date_dtype = features['Date'].dtype

    if np.issubdtype(features['Date'].dtype, np.datetime64):
        try:
            dates = np.datetime_as_string(features['Date'], unit='D')
        except AttributeError as e:
            raise TypeError(f"Error converting 'Date' to strings: {e}")

        try:
            formatted_dates = [f"{date_str[8:10]}/{date_str[5:7]}/{date_str[0:4]}" for date_str in dates]
        except Exception as e:
            raise ValueError(f"Error rearranging date strings to 'DD/MM/YYYY': {e}")
    elif date_dtype.kind in {'U', 'S'}:
        # Handle NumPy Unicode or byte strings
        formatted_dates = features['Date']
    else:
        date_dtype = features['Date'].dtype
        raise TypeError(f"Unsupported 'Date' column type: {date_dtype}. Supported types are numpy.datetime64 and string in 'DD/MM/YYYY' format.")
    
    for column in columns_to_plot:
        plx.clear_figure()
        plx.plot_size(width=100, height=20)
        plx.title(f"{column} over Time")
        plx.xlabel("Date")
        plx.ylabel(column)
        plx.plot(formatted_dates, features[column])
        plx.show()
