"""The live plotter is used to plot live data as it is being streamed."""

import time
from abc import ABC, abstractmethod
from threading import Lock, Thread

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib import animation, artist

# Default animation interval in milliseconds.
DEFAULT_ANIMATION_INTERVAL = 100  # milliseconds


class LivePlotter(ABC):
    """Interface for a live plotter."""

    def __init__(self,
                 title: str,
                 xlabel: str,
                 ylabel: str,
                 xmax: float,
                 ymin: float,
                 ymax: float,
                 num_traces: int = 1,
                 secindices: tuple[int] = None,
                 secylabel: str = None,
                 secymin: float = None,
                 secymax: float = None) -> None:
        self.num_traces = num_traces
        self.xmax = xmax

        # Prepare the plot.
        plt.style.use(["science", "grid"])
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim((0, xmax))
        self.ax.set_ylim((ymin, ymax))
        if secindices is not None:
            self.secax = self.ax.twinx()
            self.secax.set_ylabel(secylabel)
            self.secax.set_ylim((secymin, secymax))

        # Initialize the data and the traces. If the live plot is continuous,
        # the x-axis is the time in seconds. Otherwise, the x-axis is the
        # sample index.
        self.x = np.zeros(1)
        self.y = np.zeros((1, num_traces))
        self.data_lock = Lock()
        self.traces = []
        for i in range(num_traces):
            args = {
                "color": f"C{i}",
                "marker": "^",
                "label": f"Trace {i + 1}",
            }
            if secindices is None or i not in secindices:
                trace, = self.ax.plot(self.x, self.y[:, i], **args)
            else:
                trace, = self.secax.plot(self.x, self.y[:, i], **args)
            self.traces.append(trace)
        self.ax.legend(handles=self.traces)

        # Create a thread for updating the data.
        self.data_thread = Thread(target=self._update_data)

    def __del__(self) -> None:
        if self.data_thread.is_alive():
            self.data_thread.join()

    def run(self) -> None:
        """Runs the live plotter."""
        self.data_thread.start()
        self._run_animation()

    @abstractmethod
    def next_data(self) -> tuple[float, float | tuple[float]]:
        """Returns the next data to plot.

        This function blocks until the next data is available.
        """

    @abstractmethod
    def next(self) -> float | tuple[float]:
        """Returns the next y-value to plot.

        This function blocks until the next value is available.
        """

    def _update_data(self) -> None:
        """Updates the data to plot."""
        while True:
            x, y = self.next_data()
            with self.data_lock:
                self.x = np.append(self.x, x)
                self.y = np.vstack((self.y, y))

                # Remove old data.
                old = np.max(self.x) - self.x <= self.xmax
                self.x = self.x[old]
                self.x -= np.min(self.x)
                self.y = self.y[old]

    def _update_animation(self, frame: int) -> tuple[artist.Artist]:
        """Updates the animation for the next frame.

        Args:
            frame: Frame number.

        Returns:
            Iterable of artists.
        """
        with self.data_lock:
            for i in range(self.num_traces):
                self.traces[i].set_data(self.x, self.y[:, i])
        return self.traces

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

    def __init__(self,
                 max_num_points: int,
                 title: str,
                 xlabel: str,
                 ylabel: str,
                 ymin: float,
                 ymax: float,
                 num_traces: int = 1,
                 secindices: tuple[int] = None,
                 secylabel: str = None,
                 secymin: float = None,
                 secymax: float = None) -> None:
        super().__init__(title,
                         xlabel,
                         ylabel,
                         max_num_points,
                         ymin,
                         ymax,
                         num_traces=num_traces,
                         secindices=secindices,
                         secylabel=secylabel,
                         secymin=secymin,
                         secymax=secymax)

    def next_data(self) -> tuple[float, float | tuple[float]]:
        """Returns the next data to plot.

        This function blocks until the next data is available.
        """
        y = self.next()
        x = np.max(self.x) + 1
        return x, y


class ContinuousLivePlotter(LivePlotter):
    """Interface for a continuous live plotter.

    The x-axis represents the time in seconds.
    """

    def __init__(self,
                 max_duration: float,
                 title: str,
                 xlabel: str,
                 ylabel: str,
                 ymin: float,
                 ymax: float,
                 num_traces: int = 1,
                 secindices: tuple[int] = None,
                 secylabel: str = None,
                 secymin: float = None,
                 secymax: float = None) -> None:
        super().__init__(title,
                         xlabel,
                         ylabel,
                         max_duration,
                         ymin,
                         ymax,
                         num_traces=num_traces,
                         secindices=secindices,
                         secylabel=secylabel,
                         secymin=secymin,
                         secymax=secymax)
        self.last_data_time = 0

    def next_data(self) -> tuple[float, float | tuple[float]]:
        """Returns the next data to plot.

        This function blocks until the next data is available.
        """
        y = self.next()
        x = np.max(self.x) + time.time() - self.last_data_time
        self.last_data_time = time.time()
        return x, y
