import numpy as np
from typing import Tuple
from numpy.random import Generator

class RandomWalkProcess:
    """
    Symmetric random walk process with ±1 increments scaled by sqrt(dt).
    """
    def __init__(self, n_steps: int, n_paths: int, dt: float = 1.0, rng: Generator = None):
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = dt
        self.rng = rng or np.random.default_rng()

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random walk sample paths.

        Returns:
          paths: ndarray, shape (n_paths, n_steps+1)
          increments: ndarray, shape (n_paths, n_steps)
        """
        # generate ±1 increments
        increments = self.rng.choice([-1, 1], size=(self.n_paths, self.n_steps))
        scaled_inc = increments * np.sqrt(self.dt)
        # initialize paths array with starting value 0
        paths = np.zeros((self.n_paths, self.n_steps + 1), dtype=scaled_inc.dtype)
        # cumulative sum of scaled increments
        paths[:, 1:] = np.cumsum(scaled_inc, axis=1)
        return paths, increments
