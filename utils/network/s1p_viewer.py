"""The s1p viewer visualizes the S11 parameters of a 1-port network."""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import skrf as rf


class S1PViewer:
    """1-port network viewer."""

    def __init__(self, file: str) -> None:
        self.network = rf.Network(file)

    @property
    def frequency_axis(self) -> np.ndarray:
        """Frequency axis in Hz."""
        return self.network.f

    @property
    def s11_parameters(self) -> np.ndarray:
        """S11 parameters."""
        return np.squeeze(self.network.s)

    @property
    def z_parameters(self) -> np.ndarray:
        """Z parameters."""
        return np.squeeze(self.network.z)

    def plot_s11_magnitude(self) -> None:
        """Plots the S11 magnitude."""
        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(figsize=(12, 8))
        self.network.plot_s_db10(ax=ax, label="S11 magnitude")
        ax.set_title("S11 magnitude")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("S11 magnitude [dB]")
        plt.legend()
        plt.show()

    def plot_z(self) -> None:
        """Plots the real and imaginary components of the Z parameters."""
        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(figsize=(12, 8))
        self.network.plot_z_re(ax=ax, label="Real component")
        self.network.plot_z_im(ax=ax, label="Imaginary component")
        ax.set_title("Z magnitude")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Z magnitude [dB]")
        plt.legend()
        plt.show()
