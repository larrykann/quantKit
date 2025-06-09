"""quantKit.financial.returns

Vectorized return calculation functions optimized for performance.
All functions operate on numpy arrays for maximum speed.
"""

import numpy as np

__all__ = [
    "simple_returns",
    "log_returns", 
    "multi_period_simple_returns",
    "multi_period_log_returns",
]


def simple_returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute simple returns from price series using vectorized operations.
    
    Formula: R_t = (S_t - S_{t-periods}) / S_{t-periods}
    
    Args:
        prices: 1D array of asset prices
        periods: Number of periods for return calculation (default=1)
        
    Returns:
        Simple returns, same length as input with first `periods` elements as NaN
        
    Examples:
        >>> prices = np.array([100., 105., 110., 108.])
        >>> simple_returns(prices)
        array([nan, 0.05, 0.04761905, -0.01818182])
    """
    if prices.ndim != 1:
        raise ValueError("prices must be 1D array")
    if len(prices) <= periods:
        raise ValueError(f"prices length {len(prices)} must be > periods {periods}")
    
    returns = np.empty_like(prices, dtype=np.float64)
    returns[:periods] = np.nan
    
    # Vectorized calculation: (prices[periods:] / prices[:-periods]) - 1
    returns[periods:] = (prices[periods:] / prices[:-periods]) - 1.0
    
    return returns


def log_returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute logarithmic returns from price series using vectorized operations.
    
    Formula: r_t = log(S_t / S_{t-periods}) = log(S_t) - log(S_{t-periods})
    
    Args:
        prices: 1D array of asset prices (must be positive)
        periods: Number of periods for return calculation (default=1)
        
    Returns:
        Log returns, same length as input with first `periods` elements as NaN
        
    Examples:
        >>> prices = np.array([100., 105., 110., 108.])
        >>> log_returns(prices)
        array([nan, 0.04879016, 0.04652002, -0.01834786])
    """
    if prices.ndim != 1:
        raise ValueError("prices must be 1D array")
    if len(prices) <= periods:
        raise ValueError(f"prices length {len(prices)} must be > periods {periods}")
    if np.any(prices <= 0):
        raise ValueError("all prices must be positive for log returns")
    
    returns = np.empty_like(prices, dtype=np.float64)
    returns[:periods] = np.nan
    
    # Vectorized calculation: log(prices[periods:] / prices[:-periods])
    returns[periods:] = np.log(prices[periods:] / prices[:-periods])
    
    return returns


def multi_period_simple_returns(
    single_period_returns: np.ndarray, 
    periods: int
) -> np.ndarray:
    """Compute multi-period simple returns from single-period returns.
    
    Formula: R_t(τ) = ∏(1 + R_{t-i}) - 1 for i = 0 to τ-1
    
    Args:
        single_period_returns: 1D array of single-period simple returns
        periods: Number of periods to compound
        
    Returns:
        Multi-period returns, same length as input with first `periods-1` elements as NaN
        
    Examples:
        >>> returns = np.array([np.nan, 0.05, 0.03, -0.02, 0.01])
        >>> multi_period_simple_returns(returns, 3)
        array([nan, nan, nan, 0.05909, 0.0192072])
    """
    if single_period_returns.ndim != 1:
        raise ValueError("returns must be 1D array")
    if len(single_period_returns) < periods:
        raise ValueError(f"returns length {len(single_period_returns)} must be >= periods {periods}")
    
    n = len(single_period_returns)
    multi_returns = np.empty(n, dtype=np.float64)
    multi_returns[:periods-1] = np.nan
    
    # Convert returns to gross returns (1 + R)
    gross_returns = 1.0 + single_period_returns
    
    # Use rolling window product for efficiency
    for i in range(periods - 1, n):
        window = gross_returns[i - periods + 1:i + 1]
        if np.any(np.isnan(window)):
            multi_returns[i] = np.nan
        else:
            multi_returns[i] = np.prod(window) - 1.0
    
    return multi_returns


def multi_period_log_returns(
    single_period_log_returns: np.ndarray, 
    periods: int
) -> np.ndarray:
    """Compute multi-period log returns from single-period log returns.
    
    Formula: r_t(τ) = Σr_{t-i} for i = 0 to τ-1
    
    Args:
        single_period_log_returns: 1D array of single-period log returns
        periods: Number of periods to sum
        
    Returns:
        Multi-period log returns, same length as input with first `periods-1` elements as NaN
        
    Examples:
        >>> log_rets = np.array([np.nan, 0.04879, 0.02956, -0.02020, 0.00995])
        >>> multi_period_log_returns(log_rets, 3)
        array([nan, nan, nan, 0.05815, 0.00931])
    """
    if single_period_log_returns.ndim != 1:
        raise ValueError("returns must be 1D array")
    if len(single_period_log_returns) < periods:
        raise ValueError(f"returns length {len(single_period_log_returns)} must be >= periods {periods}")
    
    n = len(single_period_log_returns)
    multi_returns = np.empty(n, dtype=np.float64)
    multi_returns[:periods-1] = np.nan
    
    # Use rolling window sum for efficiency
    for i in range(periods - 1, n):
        window = single_period_log_returns[i - periods + 1:i + 1]
        if np.any(np.isnan(window)):
            multi_returns[i] = np.nan
        else:
            multi_returns[i] = np.sum(window)
    
    return multi_returns
