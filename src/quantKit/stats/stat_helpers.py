"""
Statistical Helper Functions

Table of Contents:
    - atr(values: np.ndarray) -> array
    - compute_serial_correlated_break(values: np.ndarray, ncases: int, min_recent: int, max_recent: int, lag: int) -> tuple
    - fast_exponential_smoothing(values: np.ndarray) -> float
    - iqr(values: np.ndarray) -> float
    - mutual_info(feature: np.ndarray, target: np.ndarray, nbins_feature: int = 10, nbins_target: int = 10) -> float
    - normal_cdf(z: float) -> float
    - populate_contingency_matrix(feature: np.ndarray, target: np.ndarray, nbins_feature: int, nbins_target: int) -> tuple
    - range_iqr_ratio(values: np.ndarray, iqr: float) -> float
    - relative_entropy(p: np.ndarray, q: np.ndarray) -> float
    - simple_stats(values: np.ndarray) -> tuple[int, float, float, float]
    - u_test(n1: int, x1: np.ndarray, n2: int, x2: np.ndarray) -> tuple

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

# ---------------------------------------------
# Serial-Correlated Mean Break Test Function
# ---------------------------------------------
def compute_serial_correlated_break(
    values: np.ndarray, 
    ncases: int, 
    min_recent: int, 
    max_recent: int, 
    lag: int
) -> tuple:
    """
    Serial-Correlated Mean Break Test

    This test uses the `u_test` function to check a serial-correlated data series
    for a break in its mean. It is based on the methods found in Timothy Masters'
    "Statistically Sound Indicators for Financial Market Prediction".

    Parameters:
    - values (np.ndarray): The array containing the data series to be tested.
    - ncases (int): The total number of observations in the dataset.
    - min_recent (int): The minimum recent history cases to consider in the test (bars, observations).
    - max_recent (int): The maximum recent history cases to consider in the test (bars, observations).
    - lag (int): The maximum extent of serial correlation considered; this is typically the look-back period of the indicator.

    Returns:
    - tuple: (max_crit, ibreak)
        - max_crit (float): The maximum U statistic observed across all tested boundaries,
                           indicating the strongest evidence of a mean break found in the data series.
        - ibreak (int): The boundary (in terms of number of observations from the start of the series) at which the maximum U statistic was observed,
                      suggesting the most likely position for a mean break.
    """
    max_crit = -np.inf
    ibreak = -1
    for offset in range(lag + 1):
        for nrecent in range(min_recent, max_recent + 1, lag):
            if nrecent < offset + 1:
                continue
            n1 = (nrecent - offset - 1) // lag + 1
            n2 = ncases - n1
            if n2 < 1:
                continue
            x1 = values[:n1]
            x2 = values[n1:n1+n2]
            u_stat, crit = u_test(n1, x1, n2, x2)
            if abs(crit) > max_crit:
                max_crit = abs(crit)
                ibreak = nrecent

    return max_crit, ibreak

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

# ---------------------------------------------
# Mann-Whitney U Test Function
# ---------------------------------------------
def u_test(n1: int, x1: np.ndarray, n2: int, x2: np.ndarray) -> tuple:
    """
    Mann-Whitney U-test.

    This test returns the U statistic based on set 1 relative to set 2, allowing for a one-tailed test.
    A small U indicates that the mean of set 1 is greater than that of set 2.
    Note that U' = n1 * n2 - U. Most statistical tables perform a two-tailed test by
    considering the smaller of U and U'.

    This function also computes the normal approximation z-score for the one-tailed test,
    which is accurate when n1 + n2 > 20. It flips the sign of z so that z > 0 when mean1 > mean2.

    Parameters:
    - n1 (int): Number of elements in sample 1
    - x1 (np.ndarray): Sample 1 data
    - n2 (int): Number of elements in sample 2
    - x2 (np.ndarray): Sample 2 data

    Returns:
    - tuple: (U, z)
        - U (float): The U statistic
        - z (float): The z-score for the normal approximation
    """
    # Combine the data into a single array while tracking original group memberships
    combined = np.concatenate((x1, x2))
    group = np.zeros(n1 + n2, dtype=np.int32)
    group[:n1] = 1  # Group 1 for x1, 0 for x2 by default

    indices = np.argsort(combined)
    sorted_combined = combined[indices]
    sorted_group = group[indices]

    # Compute ranks and tie correction.
    ranks = np.empty(n1 + n2, dtype=np.float64)
    tie_correction = 0.0
    i = 0
    while i < n1 + n2:
        start = i
        while i < n1 + n2 - 1 and sorted_combined[i] == sorted_combined[i + 1]:
            i += 1
        i += 1
        ntied = i - start
        tie_correction += ntied ** 3 - ntied
        for j in range(start, i):
            ranks[j] = start + (i - start - 1) / 2.0 + 1

    # Compute the U statistic.
    R = np.sum(ranks[sorted_group == 1])
    U = n1 * n2 + 0.5 * (n1 * (n1 + 1.0)) - R

    # Compute the normal approximation.
    dn = float(n1 + n2)
    term1 = n1 * n2 / (dn * (dn - 1.0))
    term2 = (dn**3 - dn - tie_correction) / 12.0
    z = (0.5 * n1 * n2 - U) / np.sqrt(term1 * term2)

    return U, z

