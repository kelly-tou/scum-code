"""The serial live plotter is used to plot live data from a serial port."""

from collections.abc import Callable

from absl import logging

from utils.plotter.live_plotter import ContinuousLivePlotter
from utils.serial.serial_interface import SerialInterface


class SerialLivePlotter(ContinuousLivePlotter):
    """Serial live plotter to plot data from a serial port."""

    def __init__(self,
                 port: str,
                 baudrate: int,
                 max_duration: float,
                 parse_data: Callable[[str], float | tuple[float]],
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
        super().__init__(max_duration,
                         title,
                         xlabel,
                         ylabel,
                         ymin,
                         ymax,
                         num_traces=num_traces,
                         secindices=secindices,
                         secylabel=secylabel,
                         secymin=secymin,
                         secymax=secymax)
        self.parse_data = parse_data

        # Open the serial port.
        self.serial = SerialInterface(port, baudrate)

    def next(self) -> float | tuple[float]:
        """Returns the next y-value to plot.

        This function blocks until the next value is available.
        """
        read_data = bytes()
        while len(read_data) == 0:
            read_data = self.serial.read()
            try:
                read_data = read_data.decode().strip()
            except:
                logging.error("Failed to decode read data.")
                read_data = bytes()
        return self.parse_data(read_data)
