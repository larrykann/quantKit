from typing import Tuple
import numpy as np
import math

def opt_thresh(
    min_cases_percent: float,
    predictor: np.ndarray,
    target: np.ndarray,
    use_log: bool = False
) -> Tuple[float, float, float, float, float]:
    """
    Optimize thresholds to maximize profit factors for long and short positions.

    Parameters
    ----------
    min_cases_percent : float
        Minimum percentage of cases for threshold calculation.
    predictor : numpy.ndarray
        Array of predictor values (signal values).
    target : numpy.ndarray
        Array of target values (returns).
    use_log : bool, optional
        Whether to use logarithmic values (default is False).

    Returns
    -------
    numpy.ndarray
        Array containing:
            - Overall profit factor (`pf_all`)
            - Best long threshold (`high_thresh`)
            - Best long profit factor (`pf_long`)
            - Best short threshold (`low_thresh`)
            - Best short profit factor (`pf_short`)

    Raises
    ------
    ValueError
        If `min_cases_percent` is less than 0 or greater than 100.
        If `predictor` and `target` arrays have different lengths.

    Example
    -------
    ```python
    import numpy as np
    from your_library_name.stats.optimization import opt_thresh

    # Example data
    np.random.seed(42)  # For reproducibility
    predictor = np.random.rand(100)
    target = np.random.randn(100)

    # Optimize thresholds with a minimum of 5% cases
    results = opt_thresh(min_cases_percent=5, predictor=predictor, target=target, use_log=True)

    # Extract results
    pf_all, high_thresh, pf_long, low_thresh, pf_short = results

    print(f"Overall Profit Factor: {pf_all}")
    print(f"Best Long Threshold: {high_thresh}, Profit Factor: {pf_long}")
    print(f"Best Short Threshold: {low_thresh}, Profit Factor: {pf_short}")
    ```
    """
    n = len(predictor)

    if not (0 <= min_cases_percent <= 100):
        raise ValueError("min_cases_percent must be between 0 and 100.")

    if len(predictor) != len(target):
        raise ValueError("predictor and target arrays must have the same length.")

    min_kept = max(1, int(min_cases_percent * n / 100))

    # Optional: Apply logarithmic transformation to returns if use_log is True
    if use_log:
        with np.errstate(divide='ignore'):
            target = np.log(target + 1)  # Adjust as needed for your use case

    # Initialize 'above' wins and losses with total sums
    win_above = np.sum(target[target > 0])
    lose_above = -np.sum(target[target <= 0])

    pf_all = win_above / (lose_above + 1e-30)
    best_high_pf = pf_all
    best_high_index = 0  # Threshold at smallest value implies complete set

    # Initialize 'below' wins and losses
    win_below = 0.0
    lose_below = 0.0
    best_low_pf = -1.0
    best_low_index = n - 1  # Placeholder index

    # Sort predictor and align target accordingly
    sorted_indices = np.argsort(predictor)
    work_signal = predictor[sorted_indices]
    work_return = target[sorted_indices]

    for i in range(n - 1):
        # Remove current return from 'above' set
        if work_return[i] > 0.0:
            win_above -= work_return[i]
        else:
            lose_above += work_return[i]  # Note: lose_above is negative

        # Add current return to 'below' set
        if work_return[i] > 0.0:
            lose_below += work_return[i]
        else:
            win_below -= work_return[i]

        # Skip if the next signal value is the same (no new threshold)
        if work_signal[i + 1] == work_signal[i]:
            continue

        # Check for 'above' set
        if (n - i - 1) >= min_kept:
            current_pf_high = win_above / (lose_above + 1e-30)
            if current_pf_high > best_high_pf:
                best_high_pf = current_pf_high
                best_high_index = i + 1

        # Check for 'below' set
        if (i + 1) >= min_kept:
            current_pf_low = win_below / (lose_below + 1e-30)
            if current_pf_low > best_low_pf:
                best_low_pf = current_pf_low
                best_low_index = i + 1

    high_thresh = work_signal[best_high_index]
    low_thresh = work_signal[best_low_index]
    pf_high = best_high_pf
    pf_low = best_low_pf


    return (pf_all, high_thresh, pf_high, low_thresh, pf_low)

