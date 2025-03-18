from typing import Tuple
import numpy as np
from pyQuantTools.stats.threshold.threshold_opt import opt_thresh

def opt_MCPT(
    signal_vals: np.ndarray,
    returns: np.ndarray,
    min_kept: int,
    flip_sign: bool,
    nreps: int
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Perform Monte Carlo Permutation Test (MCPT) for optimal threshold evaluation.
    
    Parameters
    ----------
    signal_vals : numpy.ndarray
        Array of signal (indicator) values.
    returns : numpy.ndarray
        Array of corresponding return values.
    min_kept : int
        Minimum number of cases to keep for threshold calculation.
    flip_sign : bool
        If True, flips the sign of the signal values.
    nreps : int
        Number of replications for the permutation test, including the original data.
        Must be non-negative. If `nreps` is 0, the function will not perform any permutations
        and p-values will not be calculated.
    
    Returns
    -------
    Tuple[float, float, float, float, float, float, float, float]
        A tuple containing:
            - pf_all (float): Profit factor of the entire dataset.
            - high_thresh (float): Optimal threshold for long trades.
            - pf_high (float): Profit factor for long trades above the threshold.
            - low_thresh (float): Optimal threshold for short trades.
            - pf_low (float): Profit factor for short trades below the threshold.
            - pval_long (float): P-value for the long trades' profit factor.
            - pval_short (float): P-value for the short trades' profit factor.
            - pval_best (float): P-value for the best-side trades' profit factor.
    
    Raises
    ------
    ValueError
        If `nreps` is negative.
        If `signal_vals` and `returns` arrays have different lengths.
    
    Example
    -------
    ```python
    import numpy as np
    from your_library_name.stats.mcpt import opt_MCPT
    
    # Example data
    np.random.seed(42)  # For reproducibility
    signal_vals = np.random.rand(100)
    returns = np.random.randn(100)
    
    # Define parameters
    min_kept = max(1, int(5 * len(signal_vals) / 100))  # 5% of 100 = 5
    flip_sign = False
    nreps = 100  # Number of permutations
    
    # Perform MCPT
    results = opt_MCPT(
        signal_vals=signal_vals,
        returns=returns,
        min_kept=min_kept,
        flip_sign=flip_sign,
        nreps=nreps
    )
    
    # Unpack results
    (
        pf_all,
        high_thresh,
        pf_high,
        low_thresh,
        pf_low,
        pval_long,
        pval_short,
        pval_best
    ) = results
    
    print(f"Overall Profit Factor: {pf_all}")
    print(f"Best Long Threshold: {high_thresh}, Profit Factor: {pf_high}, P-Value: {pval_long}")
    print(f"Best Short Threshold: {low_thresh}, Profit Factor: {pf_low}, P-Value: {pval_short}")
    print(f"Best-Side P-Value: {pval_best}")
    ```
    """
    if nreps < 0:
        raise ValueError("nreps must be non-negative.")
    
    if len(signal_vals) != len(returns):
        raise ValueError("signal_vals and returns arrays must have the same length.")
    
    n = len(signal_vals)
    
    # Initialize counts (start at 1 to include the original unshuffled data)
    long_count = 1
    short_count = 1
    best_count = 1
    
    # Copy returns into work_permute array
    work_permute = np.copy(returns)
    
    # If flip_sign is True, flip the sign of the signal values
    if flip_sign:
        signal_vals = -signal_vals
    
    # Perform the first iteration with the original data
    pf_all, high_thresh, pf_high, low_thresh, pf_low = opt_thresh(
        min_cases_percent=(min_kept / n) * 100,
        predictor=signal_vals,
        target=work_permute,
        use_log=False  # Adjust based on your requirements
    )
    
    original_best_pf = max(pf_high, pf_low)
    
    for irep in range(1, nreps):
        # Shuffle the work_permute array
        np.random.shuffle(work_permute)
        
        # Optimize thresholds with the shuffled returns
        pf_all_rep, high_thresh_rep, pf_high_rep, low_thresh_rep, pf_low_rep = opt_thresh(
            min_cases_percent=(min_kept / n) * 100,
            predictor=signal_vals,
            target=work_permute,
            use_log=False  # Adjust based on your requirements
        )
        
        best_pf_rep = max(pf_high_rep, pf_low_rep)
        
        # Update counts based on comparisons with original profit factors
        if pf_high_rep >= pf_high:
            long_count += 1
        if pf_low_rep >= pf_low:
            short_count += 1
        if best_pf_rep >= original_best_pf:
            best_count += 1
    
    # Calculate p-values
    if nreps > 0:
        pval_long = long_count / nreps
        pval_short = short_count / nreps
        pval_best = best_count / nreps
    else:
        pval_long = pval_short = pval_best = float('nan')  # Undefined p-values
    
    return (
        pf_all,
        high_thresh,
        pf_high,
        low_thresh,
        pf_low,
        pval_long,
        pval_short,
        pval_best
    )

