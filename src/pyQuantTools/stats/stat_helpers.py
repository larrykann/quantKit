"""
Statistical Helper Functions

Table of Contents:
    - calculate_iqr(values: np.ndarray) -> float
    - calculate_rnq_iqr(values: np.ndarray, iqr: float) -> float
    - relative_entropy(p: np.ndarray, q: np.ndarray) -> float
    - simple_stats(values: np.ndarray) -> tuple[int, float, float, float]

"""
import numpy as np

def iqr(values: np.ndarray) -> float:
    """
    Calculate the Interquartile Range (IQR) for a given set of values.
    
    Parameters:
    - values (NumPy array): An array of values.

    Returns:
    - float: The calculated IQR.
    """
    q1, q3 = np.percentile(values, [25, 75])
    return q3 -q1

def range_iqr_ratio(values: np.ndarray) -> float:
    """
    Calculate the range over IQR (Interquartile Range) ratio for a given set of values.
    
    Parameters:
    - values (NumPy array): An array of values.

    Returns:
    - float: The range over IQR ratio.
    """
    calculated_iqr = iqr(values)
    return (values.max() - values.min()) / (calculated_iqr + 1.e-60)

def relative_entropy(values: np.ndarray) -> float:
    """
    Calculate the entropy for a given set of values by binning them into specified bins.
    
    Parameters:
    - values (NumPy array): An array of values.

    Returns:
    - float: The calculated relative entropy.
    """
    n = len(values)

    # Determine number of bins based on the number of values
    if n >= 10000:
        nbins = 20
    elif n >= 1000:
        nbins = 10
    elif n >= 100:
        nbins = 5
    else:
        nbins = 3

    # Initialize counts
    counts = np.zeros(nbins, dtype=int)

    # Calculate min and max for normalization
    xmin = values.min()
    xmax = values.max()

    # Calculate factor to map values to bins
    factor = (nbins - 0.00000000001) / (xmax - xmin + 1.e-60)

    # Count occurrences in each bin
    for value in values:
        k = int(factor * (value - xmin))
        counts[k] += 1

    # Calculate entropy
    ent_sum = 0.0
    for count in counts:
        if count > 0:
            p = count / n
            ent_sum -= p * np.log(p)

    # Normalize by the maximum possible entropy
    return ent_sum / np.log(nbins)

def simple_stats(values: np.ndarray) -> tuple[int, float, float, float]:
    """
    Calculate simple statistics including number of cases, mean, minimum, and maximum for a given set of values.
    
    Parameters:
    - values (NumPy array): An array of values for which to calculate statistics.

    Returns:
    - tuple: A tuple containing ncases, mean, min_value, max_value.
    """
    ncases = values.size
    mean = np.mean(values)
    min_value = np.min(values)
    max_value = np.max(values)

    return ncases, mean, min_value, max_value
