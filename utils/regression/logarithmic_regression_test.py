import numpy as np
from absl.testing import absltest

from utils.regression.logarithmic_regression import LogarithmicRegression


class LogarithmicRegressionTestCase(absltest.TestCase):

    def test_logarithm(self):
        x = np.array([1, 4, 10, 15])
        y = 5 * np.log10(x) + 3
        logarithmic_regression = LogarithmicRegression(x, y, base=10)
        self.assertIsNone(
            np.testing.assert_allclose(logarithmic_regression.a, 5))
        self.assertIsNone(
            np.testing.assert_allclose(logarithmic_regression.b, 3))


if __name__ == "__main__":
    absltest.main()
