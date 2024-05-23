import numpy as np
from absl.testing import absltest

from utils.regression.parabolic_regression import ParabolicRegression


class ParabolicRegressionTestCase(absltest.TestCase):

    def test_constant(self):
        x = np.array([1, 2, 5])
        y = np.array([3, 3, 3])
        parabolic_regression = ParabolicRegression(x, y)
        self.assertIsNone(
            np.testing.assert_allclose(parabolic_regression.a, 0, atol=1e-12))
        self.assertIsNone(
            np.testing.assert_allclose(parabolic_regression.b, 0, atol=1e-12))
        self.assertIsNone(np.testing.assert_allclose(parabolic_regression.c, 3))

    def test_parabola(self):
        x = np.array([1, 2, 3])
        y = np.array([5, 10, 17])
        parabolic_regression = ParabolicRegression(x, y)
        self.assertIsNone(np.testing.assert_allclose(parabolic_regression.a, 1))
        self.assertIsNone(np.testing.assert_allclose(parabolic_regression.b, 2))
        self.assertIsNone(np.testing.assert_allclose(parabolic_regression.c, 2))


if __name__ == "__main__":
    absltest.main()
