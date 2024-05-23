import numpy as np
from absl.testing import absltest

from utils.regression.linear_regression import (LinearRegression,
                                                WeightedLinearRegression)


class LinearRegressionTestCase(absltest.TestCase):

    def test_constant(self):
        x = np.array([1, 2])
        y = np.array([3, 3])
        linear_regression = LinearRegression(x, y)
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.slope, 0, atol=1e-12))
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.y_intercept, 3))

    def test_constant_with_error(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 1])
        linear_regression = LinearRegression(x, y)
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.slope, 0, atol=1e-12))
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.y_intercept, 2))

    def test_line(self):
        x = np.array([1, 4])
        y = np.array([3, 9])
        linear_regression = LinearRegression(x, y)
        self.assertIsNone(np.testing.assert_allclose(linear_regression.slope,
                                                     2))
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.y_intercept, 1))

    def test_line_with_error(self):
        x = np.array([1, 4, 7])
        y = np.array([2, 11, 14])
        linear_regression = LinearRegression(x, y)
        self.assertIsNone(np.testing.assert_allclose(linear_regression.slope,
                                                     2))
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.y_intercept, 1))


class WeightedLinearRegressionTestCase(absltest.TestCase):

    def test_constant(self):
        x = np.array([1, 2])
        y = np.array([3, 3])
        weights = np.array([1, 2])
        linear_regression = WeightedLinearRegression(x, y, weights)
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.slope, 0, atol=1e-12))
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.y_intercept, 3))

    def test_constant_with_error(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 1])
        weights = np.array([0.25, 1, 0.25])
        linear_regression = WeightedLinearRegression(x, y, weights)
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.slope, 0, atol=1e-12))
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.y_intercept, 3))

    def test_line(self):
        x = np.array([1, 4])
        y = np.array([3, 9])
        weights = np.array([1, 2])
        linear_regression = WeightedLinearRegression(x, y, weights)
        self.assertIsNone(np.testing.assert_allclose(linear_regression.slope,
                                                     2))
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.y_intercept, 1))

    def test_line_with_error(self):
        x = np.array([1, 4, 7])
        y = np.array([3, 9, 100])
        weights = np.array([1, 1, 0])
        linear_regression = WeightedLinearRegression(x, y, weights)
        self.assertIsNone(np.testing.assert_allclose(linear_regression.slope,
                                                     2))
        self.assertIsNone(
            np.testing.assert_allclose(linear_regression.y_intercept, 1))


if __name__ == "__main__":
    absltest.main()
