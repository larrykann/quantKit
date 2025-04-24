"""quantKit.stochastics.utils

Pure-math utility and helper functions for the stochastics package in quantKit.
"""

__all__ = [
    "normal_cdf",
    "quadratic_variation",
]

import numpy as np

def normal_cdf(z: float) -> float:
    """
    Calculate the Normal Cumulative Distribution Function (CDF).

    Parameters:
    - z (float): The input value for which to calculate the CDF.

    Returns:
    - float: The calculated CDF value.

    Examples:
        >>> from quantKit.stochastics.utils import normal_cdf
        >>> normal_cdf(0.0)
        0.5
        >>> normal_cdf(1.96)
        0.9750021048517795
    """
    zz = abs(z)
    pdf = np.exp(-0.5 * zz * zz) / np.sqrt(2.0 * np.pi)
    t = 1.0 / (1.0 + zz * 0.2316419)
    poly = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t -
             0.356563782) * t + 0.319381530) * t
    return 1.0 - pdf * poly if z > 0.0 else pdf * poly

def quadratic_variation(path: np.ndarray) -> float:
    """
    Compute the quadratic variation of a single path.

    Parameters:
    - path: 1D array of shape (n_steps+1,)

    Returns:
    - float: sum of squared increments
    
    Examples:
        >>> from quantKit.stochastics.utils import quadratic_variation
        >>> import numpy as np
        >>> path = np.array([0.0, 1.0, 0.5, 1.5])
        >>> quadratic_variation(path)
        1.75
    """
    diffs = np.diff(path)
    return np.sum(diffs**2)
