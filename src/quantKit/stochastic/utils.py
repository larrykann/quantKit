"""
Pure-math utility and helper functions for the stochastics package in quantKit.

TOC
- normal_cdf
"""
import numpy as np

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


