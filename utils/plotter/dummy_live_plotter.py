"""The dummy live plotter plots a stairstep line."""

import time

from utils.plotter.live_plotter import DiscreteLivePlotter


class DummyLivePlotter(DiscreteLivePlotter):
    """Dummy live plotter to plot dummy data."""

    def __init__(self, max_num_points: int, max_counter: int) -> None:
        super().__init__(
            max_num_points,
            title="Dummy Data",
            xlabel="Index",
            ylabel="Counter",
            ymin=0,
            ymax=max_counter,
        )
        self.counter = 0
        self.max_counter = max_counter

    def next(self) -> float:
        """Returns the next data to plot.

        This function blocks until the next data is available.
        """
        time.sleep(0.5)
        self.counter = (self.counter + 1) % self.max_counter
        return self.counter
