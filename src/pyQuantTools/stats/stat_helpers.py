"""
Statistical Helper Functions

Table of Contents:
    - atr(values: np.ndarray) -> array
    - fast_exponential_smoothing(values: np.ndarray) -> float
    - iqr(values: np.ndarray) -> float
    - mutual_info(feature: np.ndarray, target: np.ndarray, nbins_feature: int = 10, nbins_target: int = 10) -> float
    - normal_cdf(z: float) -> float
    - populate_contingency_matrix(feature: np.ndarray, target: np.ndarray, nbins_feature: int, nbins_target: int) -> tuple
    - range_iqr_ratio(values: np.ndarray, iqr: float) -> float
    - relative_entropy(p: np.ndarray, q: np.ndarray) -> float
    - simple_stats(values: np.ndarray) -> tuple[int, float, float, float]

"""
import numpy as np

#--------------------
# Average True Range
#--------------------
def atr(
        high_prices: np.ndarray, 
        low_prices: np.ndarray, 
        close_prices: np.ndarray, 
        period: int = 252, 
        use_log: bool = True) -> np.ndarray:
    """
    Calculate the Average True Range (ATR), which is a measure of volatility.

    The ATR is calculated using either the logarithmic or arithmetic difference between
    the highest, lowest, and closing prices over a specified period. The logarithmic option
    is often used to better account for percentage changes in price.

    Parameters:
    - high_prices (np.ndarray): Array of high prices for the period.
    - low_prices (np.ndarray): Array of low prices for the period.
    - close_prices (np.ndarray): Array of close prices for the period.
    - period (int): The lookback period over which to calculate the ATR. Default is 252.
    - use_log (bool): If True, the ATR is calculated using logarithmic price changes.
                      If False, arithmetic differences are used. Default is True.

    Returns:
    - np.ndarray: An array of ATR values for the entire series, with NaN for the initial period.
    """
    if use_log:
        high_log = np.log(high_prices[1:] / high_prices[:-1])
        low_log = np.log(low_prices[1:] / low_prices[:-1])
        close_log = np.log(close_prices[1:] / close_prices[:-1])

        tr1 = high_log - low_log
        tr2 = np.abs(high_log - close_log)
        tr3 = np.abs(low_log - close_log)
    else:
        tr1 = high_prices[1:] - low_prices[1:]
        tr2 = np.abs(high_prices[1:] - close_prices[:-1])
        tr3 = np.abs(low_prices[1:] - close_prices[:-1])

    true_ranges = np.maximum(np.maximum(tr1, tr2), tr3)

    atr_values = np.full_like(high_prices, np.nan)
    for i in range(period, len(high_prices)):
        atr_values[i] = np.mean(true_ranges[i - period + 1:i + 1])

    return atr_values

def fast_exponential_smoothing(
        values: np.ndarray, 
        alpha: float = 0.33333333) -> np.ndarray:
    """
    Apply exponential smoothing to an array of values.
    This is not an EMA, but instead a simple exponential smoothing that applies to an entire array.

    Parameters:
    - values (np.ndarray): Array of values to smooth.
    - alpha (float): Smoothing factor (default is 0.33333333).

    Returns:
    - np.ndarray: Array of smoothed values.
    """
    smoothed_values = np.zeros_like(values)
    smoothed_values[0] = values[0]

    for i in range(1, len(values)):
        smoothed_values[i] = alpha * values[i] + (1 - alpha) * smoothed_values[i - 1]

    return smoothed_values

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

# ---------------------------------------------
# Mutual Information Function
# ---------------------------------------------
def mutual_info(
    feature: np.ndarray, 
    target: np.ndarray, 
    nbins_feature: int = 10, 
    nbins_target: int = 10
) -> float:
    """
    Calculate Mutual Information between a feature and target, including discretization.
   
    Parameters:
    - feature (np.ndarray): Data array for the feature variable.
    - target (np.ndarray): Data array for the target variable.
    - nbins_feature (int): Number of bins for the feature variables.
    - nbins_target (int): Number of bins for the target variable.

    Returns:
    - float: Calculated mutual information value for the feature-target pair.
    """
    if not isinstance(feature, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("'feature' and 'target' must both be numpy arrays.")

    feature = feature.flatten()
    target = target.flatten()
    c_xy, c_feature, c_target = populate_contingency_matrix(
        feature, target, nbins_feature, nbins_target
    )

    p_xy = c_xy / np.sum(c_xy)
    p_feature = c_feature / np.sum(c_feature)
    p_target = c_target / np.sum(c_target)

    # Avoiding log(0) by adding a small epsilon where p_xy or the denominator is zero
    p_xy_safe = np.where(p_xy > 0, p_xy, 1e-10)
    denominator = p_feature[:, None] * p_target[None, :]
    denominator_safe = np.where(denominator > 0, denominator, 1e-10)

    # Vectorized calculation of mutual information
    MI = np.sum(p_xy_safe * np.log(p_xy_safe / denominator_safe))

    return MI

def normal_cdf(z: float) -> float:
    """
    Calculate the Normal Cumulative Distribution Function (CDF).

    Parameters:
    - z (float): The input value for which to calculate the CDF.

    Returns:
    - float: The calculated CDF value.
    """
    zz = abs(z)
    pdf = np.exp(-0.5 * zz * zz) / np.sqrt(2.0 * np.pi)
    t = 1.0 / (1.0 + zz * 0.2316419)
    poly = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t -
             0.356563782) * t + 0.319381530) * t
    return 1.0 - pdf * poly if z > 0.0 else pdf * poly

# ---------------------------------------------
# Populate Contingency Matrix Function
# ---------------------------------------------
def populate_contingency_matrix(
    feature: np.ndarray, 
    target: np.ndarray, 
    nbins_feature: int, 
    nbins_target: int
) -> tuple:
    """
    Populate a contingency matrix and calculate marginal counts for the feature and target.

    Parameters:
    - feature (np.ndarray): 1D numpy array representing the feature variable.
    - target (np.ndarray): 1D numpy array representing the target variable.
    - nbins_feature (int): Number of bins for the feature.
    - nbins_target (int): Number of bins for the target.

    Returns:
    - tuple: (c_xy, c_feature, c_target) where:
        - c_xy: Contingency matrix.
        - c_feature: Marginal counts for the feature.
        - c_target: Marginal counts for the target.
    """
    if not isinstance(feature, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("'feature' and 'target' must both be numpy arrays.")

    bins_feature = np.linspace(np.min(feature), np.max(feature), nbins_feature + 1)
    bins_target = np.linspace(np.min(target), np.max(target), nbins_target + 1)

    c_xy = np.zeros((nbins_feature, nbins_target), dtype=np.float64)
    c_feature = np.zeros(nbins_feature, dtype=np.float64)
    c_target = np.zeros(nbins_target, dtype=np.float64)

    for i in range(len(feature)):
        value = feature[i]
        target_value = target[i]

        idx_feature = np.searchsorted(bins_feature, value, side='right') - 1
        idx_target = np.searchsorted(bins_target, target_value, side='right') - 1

        # Ensure idx_feature is within valid range
        if idx_feature < 0:
            idx_feature = 0
        elif idx_feature >= nbins_feature:
            idx_feature = nbins_feature - 1

        # Ensure idx_target is within valid range
        if idx_target < 0:
            idx_target = 0
        elif idx_target >= nbins_target:
            idx_target = nbins_target - 1

        c_feature[idx_feature] += 1
        c_target[idx_target] += 1   
        c_xy[idx_feature, idx_target] += 1

    return c_xy, c_feature, c_target

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
