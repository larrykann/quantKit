# quantengine/stochastic/distributions.py

"""
Vectorized implementations of core random variate generators using algorithms 
from stochastic finance (Nag, 2024).

All functions accept an `rng: numpy.random.Generator` parameter, which 
encapsulates the random number stream (seedable and reproducible).
"""
import numpy as np
from numpy.random import Generator
from typing import Callable, Sequence, Union, Tuple
from numba import njit

Size = Union[int, Tuple[int, ...]]


@njit
def _poisson_one(lam: float, u: float) -> int:
    # Devroye's sequential-search Poisson for a single uniform variate u
    x = 0
    p = np.exp(-lam)
    s = p
    while u > s:
        x += 1
        p = p * lam / x
        s += p
    return x


def uniform_rvs(rng: Generator, size: Size) -> np.ndarray:
    """
    Generate Uniform[0,1) random variates.
    """
    return rng.random(size)


def exponential_rvs(rng: Generator, lam: float, size: Size) -> np.ndarray:
    """
    Generate exponential variates with rate `lam` via the inverse-transform method:
      X = -ln(1 - U) / lam
    """
    u = rng.random(size)
    return -np.log1p(-u) / lam


def poisson_rvs(rng: Generator, lam: float, size: Size) -> np.ndarray:
    """
    Generate Poisson(λ) variates using a JIT-compiled sequential-search for the inner loop.
    """
    u = rng.random(size)
    flat_u = u.ravel()
    out = np.empty_like(flat_u, dtype=np.int64)
    for i in range(flat_u.size):
        out[i] = _poisson_one(lam, flat_u[i])
    return out.reshape(u.shape)


def discrete_rvs(rng: Generator, probabilities: np.ndarray, size: Size) -> np.ndarray:
    """
    Generate discrete variates over indices [0..len(probabilities)-1] via inverse-transform.
    """
    cum = np.cumsum(probabilities)
    u = rng.random(size)
    return np.searchsorted(cum, u)


def inverse_transform_rvs(
    rng: Generator,
    cdf_inverse: Callable[[np.ndarray], np.ndarray],
    size: Size
) -> np.ndarray:
    """
    Generic inverse-transform sampler: X = F^{-1}(U) for U~Uniform[0,1).
    """
    u = rng.random(size)
    return cdf_inverse(u)


def rejection_rvs(
    rng: Generator,
    pdf: Callable[[np.ndarray], np.ndarray],
    proposal_sampler: Callable[[int], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
    C: float,
    size: int,
    oversample_factor: float = 2.0
) -> np.ndarray:
    """
    Acceptance–rejection sampling for continuous densities:
      - pdf(x): target density, vectorized
      - proposal_sampler(n): draws n from proposal
      - proposal_pdf(x): proposal density at x
      - C: envelope constant (pdf <= C * proposal_pdf)
    Returns exactly `size` accepted samples.
    """
    samples = []
    needed = size
    while needed > 0:
        n = int(needed * oversample_factor)
        x = proposal_sampler(n)
        u = rng.random(n)
        accept = u < (pdf(x) / (C * proposal_pdf(x)))
        accepted = x[accept]
        if accepted.size:
            samples.append(accepted)
            needed -= accepted.size
    result = np.concatenate(samples)
    return result[:size]
