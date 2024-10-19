"""
Statistical Helper Functions

Table of Contents:
    - LogReturns(values: np.ndarray) -> array

"""
import numpy as np

#--------------------
# Log Returns
#--------------------
def LogReturns(values: np.ndarray, window: int = 1) -> np.ndarray:
    """
    Calculate log returns over a specified window.

    Parameters:
    - values (np.ndarray): Array of values.
    - window (int): Number of days over which to calculate returns. Default is 1.

    Returns:
    - np.ndarray: Array of log returns with the first `window` elements as zeros.
    """
    if np.any(np.isnan(values)):
        raise ValueError("Input array 'values' must not contain NaN values.")

    single_period_returns = np.log(values[1:] / values[:-1])
    output = np.zeros(len(values))
    output[window:] = np.convolve(single_period_returns, np.ones(window), mode="valid")

    return output
