import numpy as np
from absl.testing import absltest

from utils.regression.exponential_regression import ExponentialRegression


class ExponentialRegressionTestCase(absltest.TestCase):

    def test_decaying_exponential(self):
        x = np.array([0, 1, 4, 10, 15])
        y = 10 * (1 / 2)**x + 1
        exponential_regression = ExponentialRegression(x, y)
        self.assertIsNone(
            np.testing.assert_allclose(exponential_regression.a, 10, atol=5e-5))
        self.assertIsNone(
            np.testing.assert_allclose(exponential_regression.time_constant,
                                       1 / np.log(2),
                                       atol=5e-6))
        self.assertIsNone(
            np.testing.assert_allclose(exponential_regression.offset,
                                       1,
                                       atol=5e-5))


if __name__ == "__main__":
    absltest.main()
