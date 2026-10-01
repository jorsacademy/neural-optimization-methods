import unittest

import numpy as np
from study import enumerate_regions, evaluate, maximize, network


class VerificationTests(unittest.TestCase):
    def test_interior(self):
        n = network([[1], [1], [1]], [0, -0.5, -1], [1, -2, 1], 0, [0], [1])
        r = maximize(n, 0.4)
        self.assertEqual(r["status"], "counterexample")
        self.assertAlmostEqual(r["maximum_witness"], 0.5)

    def test_safe(self):
        n = network([[1]], [0], [1], 0, [-1], [1])
        self.assertEqual(maximize(n, 1)["status"], "certified_with_tolerance")

    def test_active_inactive(self):
        for b in (-3, 3):
            n = network([[1]], [b], [2], 0.1, [-1], [1])
            r = maximize(n)
            self.assertAlmostEqual(r["maximum_witness"], enumerate_regions(n))

    def test_random_oracle(self):
        for seed in range(8):
            g = np.random.default_rng(seed)
            n = network(
                g.normal(size=(4, 2)), g.normal(size=4), g.normal(size=4), 0.2, [-1, -2], [2, 1]
            )
            r = maximize(n)
            self.assertAlmostEqual(r["upper_bound"], enumerate_regions(n), places=6)
            self.assertAlmostEqual(evaluate(n, r["witness"]), r["maximum_witness"])

    def test_degenerate_box(self):
        n = network([[1]], [0], [1], 0, [0.2], [0.2])
        self.assertAlmostEqual(maximize(n)["upper_bound"], 0.2)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            network([[1]], [0], [1], 0, [2], [1])
        with self.assertRaises(ValueError):
            network([[np.nan]], [0], [1], 0, [0], [1])

    def test_limit_validation(self):
        n = network([[1]], [0], [1], 0, [0], [1])
        with self.assertRaises(ValueError):
            maximize(n, time_limit=0)


if __name__ == "__main__":
    unittest.main()
