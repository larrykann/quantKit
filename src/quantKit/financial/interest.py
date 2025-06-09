"""quantKit.financial.interest

Vectorized interest‐timevalue functions.
"""

import numpy as np

__all__ = [
    "simple_interest",
    "discrete_compound_interest",
    "continuous_compound_interest",
]


def simple_interest(
    principal: np.ndarray,
    rate: float,
    time: np.ndarray,
) -> np.ndarray:
    """Compute future value under simple interest.

    Formula:
        A = P * (1 + r * t)

    Args:
        principal (np.ndarray): Array of principal amounts P.
        rate (float): Annual interest rate r (as a decimal).
        time (np.ndarray): Array of times t in years.

    Returns:
        np.ndarray: Future values A = P * (1 + r * t).

    Examples:
        >>> import numpy as np
        >>> from quantKit.financial.interest import simple_interest
        >>> simple_interest(np.array([100.]), 0.05, np.array([2.0]))
        array([110.])
    """
    return principal * (1 + rate * time)

def discrete_compound_interest(
    principal: np.ndarray,
    rate: float,
    time: np.ndarray,
    freq: int = 1,
) -> np.ndarray:
    """Compute future value under periodic compounding.

    Formula:
        A = P * (1 + r/m)^(m * t)

    Args:
        principal (np.ndarray): Array of P.
        rate (float): Annual rate r.
        time (np.ndarray): Array of t in years.
        freq (int): Number of compounding periods per year m.

    Returns:
        np.ndarray: Future values A = P * (1 + r/m)^(m * t).

    Examples:
        >>> from quantKit.financial.interest import discrete_compound_interest
        >>> discrete_compound_interest(np.array([100.]), 0.12, np.array([2.0]), freq=4)
        array([126.82503013])
    """
    return principal * np.power(1 + rate / freq, freq * time)

def continuous_compound_interest(
    principal: np.ndarray,
    rate: float,
    time: np.ndarray,
) -> np.ndarray:
    """Compute future value under continuous compounding.

    Formula:
        A = P * exp(r * t)

    Args:
        principal (np.ndarray): Array of P.
        rate (float): Annual rate r.
        time (np.ndarray): Array of t in years.

    Returns:
        np.ndarray: Future values A = P * exp(r * t).

    Examples:
        >>> from quantKit.financial.interest import continuous_compound_interest
        >>> continuous_compound_interest(np.array([100.]), 0.05, np.array([3.0]))
        array([116.183...])
    """
    return principal * np.exp(rate * time)
