"""The exponential regression class performs a exponential regression on the
given data.

y = a * exp(-x/tau) + b
"""

import numpy as np
import scipy.optimize
from absl import logging

from utils.regression.regression import Regression


class ExponentialRegression(Regression):
    """Performs an exponential regression."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.a = 0
        self.tau = 0
        self.b = 0
        super().__init__(x, y)

    @property
    def time_constant(self) -> float:
        """Returns the time constant."""
        return self.tau

    @property
    def offset(self) -> float:
        """Returns the offset."""
        return self.b

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluates the exponential regression at the given x-values.

        Args:
            x: x-values.

        Returns:
            The y-values corresponding to the x-values.
        """
        return self.a * np.exp(-1 / self.tau * x) + self.b

    def _perform_regression(self) -> None:
        """Performs an exponential regression.

        This function sets a, tau, and b, where y = a * exp(-x/tau) + b.
        """
        A = np.vstack([self.x, np.ones(len(self.x))]).T
        result = np.linalg.lstsq(A, np.log(self.y), rcond=None)[0]
        tau_guess, a_guess, b_guess = -1 / result[0], np.exp(result[1]), 0

        # Use an optimizer to find the optimal parameters of the exponential.
        # TODO(titan): Debug why this does not quite work for increasing
        # exponentials.
        def cost(params: np.ndarray):
            """Calculates how well the exponential fits the given data.

            Args:
                params: Three-dimensional vector consisting of (a, tau, b).

            Returns:
                The norm between the given exponential and the given data.
            """
            a, tau, b = params
            return np.linalg.norm(a * np.exp(-1 / tau * self.x) + b - self.y)

        optimization_results = scipy.optimize.minimize(
            cost,
            np.array([a_guess, tau_guess, b_guess]),
            method="Nelder-Mead",
            options={"maxiter": 10000},
        )
        if not optimization_results.success:
            logging.warning("Optimization failed with message: %s",
                            optimization_results.message)
        self.a, self.tau, self.b = optimization_results.x
