from typing import List, Tuple
import numpy as np
import math

def generate_threshold_table(
    signal_vals: np.ndarray,
    returns: np.ndarray,
    bin_count: int
) -> List[Tuple[float, float, float, float, float, float, float]]:
    """
    Generates a threshold table by sorting signal values, aligning returns, and calculating profit factors.

    Parameters
    ----------
    signal_vals : numpy.ndarray
        Array of signal values.
    returns : numpy.ndarray
        Array of return values.
    bin_count : int
        Number of bins for threshold calculation. Must be either 13 or 27.

    Returns
    -------
    List[Tuple[float, float, float, float, float, float, float]]
        A list of tuples, each containing:
            - Threshold (float)
            - Fraction Greater or Equal (float)
            v
            - Long Profit Factor Above Threshold (float)
            - Short Profit Factor Above Threshold (float)
            - Fraction Less (float)
            - Short Profit Factor Below Threshold (float)
            - Long Profit Factor Below Threshold (float)

    Raises
    ------
    ValueError
        If `bin_count` is not 13 or 27.

    Example
    -------
    ```python
    import numpy as np
    from your_library_name.stats.threshold_table import generate_threshold_table

    # Example data
    np.random.seed(42)  # For reproducibility
    signal_vals = np.random.rand(100)
    returns = np.random.randn(100)

    # Generate threshold table with 13 bins
    roc_table = generate_threshold_table(signal_vals, returns, bin_count=13)
    ```
    """
    n = len(signal_vals)

    if bin_count == 13:
        bins = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    elif bin_count == 27:
        bins = [
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99
        ]
    else:
        raise ValueError("Bins must be either 13 or 27.")

    # Sort the signal values and the corresponding returns
    indices = np.argsort(signal_vals)
    work_signal = signal_vals[indices]
    work_return = returns[indices]

    roc_table = []

    for bin_fraction in bins:
        k = int(bin_fraction * (n + 1)) - 1
        if k < 0:
            k = 0
        if k > n - 1:
            k = n - 1
        while k > 0 and work_signal[k - 1] == work_signal[k]:
            k -= 1
        if k == 0 or k == n - 1:
            continue

        win_above, lose_above, win_below, lose_below = 0.0, 0.0, 0.0, 0.0

        # Accumulate wins and losses below the threshold (short positions)
        for i in range(k):
            if work_return[i] > 0.0:
                lose_below += work_return[i]
            else:
                win_below -= work_return[i]

        # Accumulate wins and losses above the threshold (long positions)
        for i in range(k, n):
            if work_return[i] > 0.0:
                win_above += work_return[i]
            else:
                lose_above -= work_return[i]

        threshold = work_signal[k]
        frac_gtr_eq = (n - k) / n

        # Calculate profit factors
        long_pf_above = win_above / lose_above if lose_above > 0 else math.inf
        short_pf_above = lose_above / win_above if win_above > 0 else math.inf
        frac_less = k / n
        short_pf_below = win_below / lose_below if lose_below > 0 else math.inf
        long_pf_below = lose_below / win_below if win_below > 0 else math.inf

        # Append the results to the table
        roc_table.append((
            threshold,
            frac_gtr_eq,
            long_pf_above,
            short_pf_above,
            frac_less,
            short_pf_below,
            long_pf_below
        ))

    return roc_table

