"""The live plotter is used to plot live data as it is being streamed."""

import time
from abc import ABC, abstractmethod
from threading import Lock, Thread

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, artist

# Default animation interval in milliseconds.
DEFAULT_ANIMATION_INTERVAL = 100  # milliseconds


class LivePlotter(ABC):
    """Interface for a live plotter."""

    def __init__(self, title: str, xlabel: str, ylabel: str, xmax: float,
                 ymin: float, ymax: float):
        self.xmax = xmax

        # Prepare the plot.
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim((0, xmax))
        self.ax.set_ylim((ymin, ymax))

        # Initialize the data and line plot. If the live plot is continuous,
        # the x-axis is the time in seconds. Otherwise, the x-axis is the
        # sample index.
        self.data = np.zeros((2, 1))
        self.data_lock = Lock()
        self.line, = self.ax.plot(self.data)

        # Create a thread for updating the data.
        self.data_thread = Thread(target=self._update_data)

    def __del__(self):
        if self.data_thread.is_alive():
            self.data_thread.join()

    def run(self) -> None:
        """Runs the live plotter."""
        self.data_thread.start()
        self._run_animation()

    @abstractmethod
    def next_data(self) -> tuple[float]:
        """Returns the next data to plot.

        This function blocks until the next data is available.
        """

    @abstractmethod
    def next(self) -> float:
        """Returns the next y-value to plot.

        This function blocks until the next value is available.
        """

    def _update_data(self) -> None:
        """Updates the data to plot."""
        while True:
            x, y = self.next_data()
            with self.data_lock:
                self.data = np.hstack((self.data, np.array([[x], [y]])))
                self.data = self.data[:,
                                      np.max(self.data[0]) -
                                      self.data[0] <= self.xmax]
                self.data[0] -= np.min(self.data[0])

    def _update_animation(self, frame: int) -> tuple[artist.Artist]:
        """Updates the animation for the next frame.

        Args:
            frame: Frame number.

        Returns:
            Iterable of artists.
        """
        with self.data_lock:
            self.line.set_data(self.data)
        return self.line,

    def _run_animation(self) -> None:
        """Runs the animation."""
        anim = animation.FuncAnimation(self.fig,
                                       self._update_animation,
                                       interval=DEFAULT_ANIMATION_INTERVAL,
                                       blit=True)
        plt.show()


class DiscreteLivePlotter(LivePlotter):
    """Interface for a discrete live plotter.

    The x-axis represents the sample index.
    """

    def __init__(self, max_num_points: int, title: str, xlabel: str,
                 ylabel: str, ymin: float, ymax: float):
        super().__init__(title, xlabel, ylabel, max_num_points, ymin, ymax)

    def next_data(self) -> tuple[float]:
        """Returns the next data to plot.

        This function blocks until the next data is available.
        """
        y = self.next()
        x = np.max(self.data[0]) + 1
        return x, y


class ContinuousLivePlotter(LivePlotter):
    """Interface for a continuous live plotter.

    The x-axis represents the time in seconds.
    """

    def __init__(self, max_duration: float, title: str, xlabel: str,
                 ylabel: str, ymin: float, ymax: float):
        super().__init__(title, xlabel, ylabel, max_duration, ymin, ymax)
        self.last_data_time = 0

    def next_data(self) -> tuple[float]:
        """Returns the next data to plot.

        This function blocks until the next data is available.
        """
        y = self.next()
        x = np.max(self.data[0]) + time.time() - self.last_data_time
        self.last_data_time = time.time()
        return x, y
