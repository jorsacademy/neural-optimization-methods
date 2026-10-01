import unittest
from unittest.mock import patch

import numpy as np
import torch
from study import Proxy, data, evaluate, objectives, oracle, train, validate


class DualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_oracle(self):
        Z = np.array([[1.0, 1.0, 0.0, 0.0, 2.0]])
        x, nu = oracle(Z)
        np.testing.assert_allclose(x, [[1, 1]])
        f, g = objectives(torch.tensor(Z), torch.tensor(x), torch.tensor(nu))
        self.assertAlmostEqual(float(f - g), 0)

    def test_weak_duality(self):
        Z = data(1, 20)
        x, _ = oracle(Z)
        f, g = objectives(torch.tensor(Z), torch.tensor(x), torch.linspace(-5, 5, 20))
        self.assertTrue(torch.all(f >= g - 1e-8))

    def test_feasibility(self):
        Z = torch.tensor(data(1, 10), dtype=torch.float32)
        x, _ = Proxy(4)(Z)
        self.assertTrue(torch.all(x >= 0))
        torch.testing.assert_close(x.sum(1), Z[:, -1])

    def test_no_label_leakage(self):
        with patch("study.oracle", side_effect=AssertionError("label leakage")):
            train(data(1, 30), epochs=5)

    def test_gradient(self):
        Z = torch.tensor(data(1, 3), dtype=torch.float64)
        x = torch.ones(3, 4, dtype=torch.float64, requires_grad=True)
        nu = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: objectives(Z, a, b), (x, nu)))

    def test_training_and_audit(self):
        m, h = train(data(4, 60), epochs=30)
        self.assertLess(h[-1], h[0])
        evaluate(m, data(8, 20))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            validate([[0, 1, 2]])
        with self.assertRaises(ValueError):
            train(data(1), mode="unknown")


if __name__ == "__main__":
    unittest.main()
