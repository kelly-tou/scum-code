"""The logarithmic regression class performs a exponential regression on the
given data.

y = a * log(x)/log(base) + b
"""

import numpy as np
import scipy.optimize
from absl import logging

from utils.regression.regression import Regression


class LogarithmicRegression(Regression):
    """Performs a logarithmic regression."""

    def __init__(self, x: np.ndarray, y: np.ndarray, base: int = np.e):
        self.a = 0
        self.b = 0
        self.base = base
        super().__init__(x, y)

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluates the logarithmic regression at the given x-values.

        Args:
            x: x-values.

        Returns:
            The y-values corresponding to the x-values.
        """
        return self.a * np.log(x) / np.log(self.base) + self.b

    def _perform_regression(self) -> None:
        """Performs a logarithmic regression.

        This function sets a and b, where y = a * log(x) / log(base) + b.
        """
        A = np.vstack(
            [np.log(self.x) / np.log(self.base),
             np.ones(len(self.x))]).T
        result = np.linalg.lstsq(A, self.y, rcond=None)[0]
        a_guess, b_guess = result

        # Use an optimizer to find the optimal parameters of the logarithm.
        def cost(params: np.ndarray):
            """Calculates how well the logarithm fits the given data.

            Args:
                params: Three-dimensional vector consisting of (a, b).

            Returns:
                The norm between the given logarithm and the given data.
            """
            a, b = params
            return np.linalg.norm(a * np.log(self.x) / np.log(self.base) + b -
                                  self.y)

        optimization_results = scipy.optimize.minimize(
            cost,
            np.array([a_guess, b_guess]),
            method="Nelder-Mead",
            options={"maxiter": 10000},
        )
        if not optimization_results.success:
            logging.warning("Optimization failed with message: %s",
                            optimization_results.message)
        self.a, self.b = optimization_results.x
