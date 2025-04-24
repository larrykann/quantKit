import numpy as np
from typing import Tuple
from numpy.random import Generator

class BrownianMotionProcess:
    """
    Brownian motion process with drift μ and diffusion σ.
    """
    def __init__(self,
                 n_steps: int,
                 n_paths: int,
                 dt: float = 1.0,
                 mu: float = 0.0,
                 sigma: float = 1.0,
                 rng: Generator = None):
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = dt
        self.mu = mu
        self.sigma = sigma
        self.rng = rng or np.random.default_rng()

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Brownian motion sample paths.

        Returns:
          paths: ndarray, shape (n_paths, n_steps+1)
          increments: ndarray, shape (n_paths, n_steps)
        """
        # normal increments: mean mu*dt, std sigma*sqrt(dt)
        increments = self.rng.normal(
            loc=self.mu * self.dt,
            scale=self.sigma * np.sqrt(self.dt),
            size=(self.n_paths, self.n_steps)
        )
        paths = np.zeros((self.n_paths, self.n_steps + 1), dtype=increments.dtype)
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths, increments
