import unittest
import numpy as np
from quantKit.stochastic.processes.random_walk import RandomWalkProcess
from quantKit.stochastic.processes.brownian_motion import BrownianMotionProcess
from quantKit.stochastic.utils import quadratic_variation

class TestProcesses(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2025)
        self.n_steps = 10000
        self.n_paths = 100
        self.dt = 0.01

    def test_random_walk_increments(self):
        rw = RandomWalkProcess(
            n_steps=self.n_steps,
            n_paths=self.n_paths,
            dt=self.dt,
            rng=self.rng
        )
        paths, inc = rw.sample()
        # increments must be ±1
        self.assertTrue(np.all(np.isin(inc, [-1, 1])))
        # paths shape includes initial zero
        self.assertEqual(paths.shape, (self.n_paths, self.n_steps + 1))

    def test_random_walk_mean(self):
        rw = RandomWalkProcess(
            n_steps=self.n_steps,
            n_paths=1000,
            dt=self.dt,
            rng=self.rng
        )
        _, inc = rw.sample()
        # empirical mean ≈ 0
        self.assertAlmostEqual(np.mean(inc), 0.0, delta=0.01)

    def test_brownian_increments_distribution(self):
        bm = BrownianMotionProcess(
            n_steps=self.n_steps,
            n_paths=self.n_paths,
            dt=self.dt,
            mu=0.0,
            sigma=1.0,
            rng=self.rng
        )
        _, inc = bm.sample()
        # increments shape
        self.assertEqual(inc.shape, (self.n_paths, self.n_steps))
        # check mean and variance
        self.assertAlmostEqual(np.mean(inc), 0.0, delta=0.01)
        self.assertAlmostEqual(np.var(inc, ddof=0), self.dt, delta=0.01)

    def test_quadratic_variation(self):
        sigma = 2.0
        bm = BrownianMotionProcess(
            n_steps=self.n_steps,
            n_paths=1,
            dt=self.dt,
            mu=0.0,
            sigma=sigma,
            rng=self.rng
        )
        paths, _ = bm.sample()
        path = paths[0]
        qv = quadratic_variation(path)
        expected = sigma**2 * self.n_steps * self.dt
        # allow ±5% tolerance
        self.assertAlmostEqual(qv, expected, delta=expected * 0.05)

if __name__ == '__main__':
    unittest.main()
