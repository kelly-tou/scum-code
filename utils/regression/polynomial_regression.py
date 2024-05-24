"""The polynomial regression class performs a polynomial regression with the
given degree on the given data.

y = a0 + a1 * x + a2 * x^2 + ...
"""

import numpy as np

from utils.regression.regression import Regression


class PolynomialRegression(Regression):
    """Performs a polynomial regression."""

    def __init__(self, x: np.ndarray, y: np.ndarray, degree: int):
        self.degree = degree
        self.coeffs: np.ndarray = None
        self.residuals = 0
        super().__init__(x, y)

    @property
    def r_squared(self) -> float:
        """Coefficient of determination."""
        total_sum_squares = np.linalg.norm(self.y - np.mean(self.y))**2
        return 1 - self.residuals / total_sum_squares

    @property
    def coefficients(self) -> float:
        """Coefficients of the polynomial regression.

        The coefficients are sorted in increasing powers.
        """
        return self.coeffs

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluates the polynomial regression at the given x-values.

        Args:
            x: x-values.

        Returns:
            The y-values corresponding to the x-values.
        """
        powers = np.arange(len(self.coeffs))
        powers_of_x = x**powers[:, np.newaxis]
        return np.dot(self.coeffs, powers_of_x)

    def _perform_regression(self) -> None:
        """Performs a polynomial regression.

        This function sets coefficients and residuals, where coefficients are
        the polynomial coefficients.
        """
        powers = np.arange(self.degree + 1)
        A = (self.x**powers[:, np.newaxis]).T
        result, residuals = np.linalg.lstsq(A, self.y, rcond=None)[:2]
        self.coeffs = result
        self.residuals = residuals[0] if len(residuals) > 0 else 0
