# tests/test_distributions.py

import unittest
import numpy as np
from quantKit.stochastic.distributions import (
    uniform_rvs,
    exponential_rvs,
    poisson_rvs,
    discrete_rvs,
    inverse_transform_rvs,
)

class TestDistributions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2025)

    def test_uniform_rvs(self):
        u = uniform_rvs(self.rng, size=1000)
        self.assertEqual(u.shape, (1000,))
        self.assertTrue(np.all((u >= 0) & (u < 1)))

    def test_exponential_rvs_mean(self):
        for lam in [0.5, 2.0, 5.0]:
            with self.subTest(lam=lam):
                x = exponential_rvs(self.rng, lam, size=100000)
                self.assertEqual(x.shape, (100000,))
                self.assertAlmostEqual(x.mean(), 1/lam, delta=0.2)

    def test_poisson_rvs_mean(self):
        for lam in [1.0, 4.0]:
            with self.subTest(lam=lam):
                x = poisson_rvs(self.rng, lam, size=100000)
                self.assertEqual(x.shape, (100000,))
                self.assertAlmostEqual(x.mean(), lam, delta=0.1)
                self.assertTrue(issubclass(x.dtype.type, np.integer))

    def test_discrete_rvs_distribution(self):
        probs = np.array([0.2, 0.8])
        x = discrete_rvs(self.rng, probs, size=100000)
        freq = np.bincount(x, minlength=2) / x.size
        self.assertAlmostEqual(freq[0], 0.2, delta=0.02)
        self.assertAlmostEqual(freq[1], 0.8, delta=0.02)

    def test_inverse_transform_rvs_exponential(self):
        inv = lambda u: -np.log1p(-u)  # Exponential(1)
        x = inverse_transform_rvs(self.rng, inv, size=100000)
        self.assertEqual(x.shape, (100000,))
        self.assertAlmostEqual(x.mean(), 1.0, delta=0.2)

if __name__ == '__main__':
    unittest.main()
