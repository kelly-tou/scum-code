"""The linear regression class performs a linear regression on the given data.

y = mx + b
"""

import numpy as np

from utils.regression.polynomial_regression import PolynomialRegression


class LinearRegression(PolynomialRegression):
    """Performs a linear regression."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__(x, y, 1)

    @property
    def slope(self) -> float:
        """Slope of the linear regression."""
        return self.coefficients[1]

    @property
    def y_intercept(self) -> float:
        """y-intercept of the linear regression."""
        return self.coefficients[0]


class WeightedLinearRegression(LinearRegression):
    """Performs a linear regression with weighted least squares."""

    def __init__(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray):
        self.weights = np.copy(weights)
        assert np.all(self.weights >= 0)
        super().__init__(x, y)

    @property
    def r_squared(self) -> float:
        """Coefficient of determination."""
        W = np.sqrt(self.weights)
        y_weighted = self.y * W
        total_sum_squares = np.linalg.norm(y_weighted - np.mean(y_weighted))**2
        return 1 - self.residuals / total_sum_squares

    @property
    def x_mean(self):
        """Weighted mean of the x-values."""
        return np.sum(self.weights * self.x) / np.sum(self.weights)

    @property
    def y_mean(self):
        """Weighted mean of the y-values."""
        return np.sum(self.weights * self.y) / np.sum(self.weights)

    @property
    def slope_variance(self) -> float:
        """Variance of the slope of the linear regression.

        This function assumes that the weights are equal to 1/sample variance.
        """
        return 1 / np.sum(self.weights * (self.x - self.x_mean)**2)

    @property
    def y_intercept_variance(self) -> float:
        """Variance of the y-intercept of the linear regression.

        This function assumes that the weights are equal to 1/sample variance.
        """
        return 1 / np.sum(self.weights) + self.slope_variance * self.x_mean**2

    def _perform_regression(self) -> None:
        """Performs a linear regression with weighted least squares.

        This function sets m, b, and residuals, where m is the slope and b is
        the y-intercept.
        """
        A = (self.x**np.arange(2)[:, np.newaxis]).T
        W = np.sqrt(self.weights)
        A_weighted = A * W[:, np.newaxis]
        y_weighted = self.y * W
        result, residuals = np.linalg.lstsq(A_weighted, y_weighted,
                                            rcond=None)[:2]
        self.coeffs = result
        self.residuals = residuals[0] if len(residuals) > 0 else 0
