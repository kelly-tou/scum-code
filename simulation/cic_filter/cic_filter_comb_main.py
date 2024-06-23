import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from absl import app, flags

from simulation.cic_filter.cic_filter_decimator import CicFilterDecimator

FLAGS = flags.FLAGS

# Filter coefficients for the averaging filter.
FILTERS = [
    np.array([1]),
    np.array([0.25, 0.75, 0.75, 0.25]),
    np.array([0.25, 1, 1, 0.25]),
    np.array([0, 1, 1, 0]),
]


def _create_comb_filter_before_decimation(coefficients: np.ndarray,
                                          R: int) -> np.ndarray:
    """Creates the comb filter coefficients before decimation.

    Args:
        coefficients: The desired filter coefficients without zero padding.
        R: Downsampling factor.

    Returns:
        The length-(R + 1) coefficients of the comb filter.
    """
    coefficient_step = R // len(coefficients)
    comb_filter = np.repeat(coefficients, coefficient_step).astype(np.float64)
    # Scale the filter to integrate to R.
    comb_filter /= np.mean(comb_filter)
    return comb_filter


def compare_crc_filter_combs(length: int, R: int) -> None:
    """Compares different comb filters for a CIC filter stage.

    Args:
        length: Signal and FFT length.
        R: Downsampling factor.
    """
    delta = np.zeros(length)
    delta[0] = 1

    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))

    for coefficients in FILTERS:
        # Apply a comb filter to the signal.
        cic_filter_decimator = CicFilterDecimator(R=R,
                                                  N=1,
                                                  comb_filter=coefficients)
        response = cic_filter_decimator.filter(delta, downsampling=False)

        # Plot the spectrum.
        omega, response_fft_abs = (
            cic_filter_decimator.calculate_spectrum_magnitude(response, length))
        ax.plot(omega, 20 * np.log10(response_fft_abs), label=coefficients)
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("FFT magnitude [dB]")
    ax.set_xlim([0, 2 * np.pi / R])
    max_magnitude = 10 * np.log10(length)
    ax.set_ylim([max_magnitude - 10, max_magnitude + 3])
    ax.axvline(np.pi / R, linestyle="--", linewidth=2, zorder=0)
    ax.legend()
    plt.show()


def plot_comb_filter_spectrum(length: int, R: int) -> None:
    """Plots the spectrum of each comb filter.

    Args:
        length: FFT length.
        R: Downsampling factor.
    """
    # Plot the spectrum of each comb filter.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))

    for coefficients in FILTERS:
        comb_filter = _create_comb_filter_before_decimation(coefficients, R)
        omega, comb_filter_fft_abs = (
            CicFilterDecimator.calculate_spectrum_magnitude(
                comb_filter, length))
        ax.plot(omega, 20 * np.log10(comb_filter_fft_abs), label=coefficients)
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("FFT magnitude [dB]")
    ax.set_xlim([0, 2 * np.pi / R])
    max_magnitude = 10 * np.log10(length)
    ax.set_ylim([max_magnitude - 10, max_magnitude + 3])
    ax.axvline(np.pi / R, linestyle="--", linewidth=2, zorder=0)
    ax.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    compare_crc_filter_combs(FLAGS.length, FLAGS.R)
    plot_comb_filter_spectrum(FLAGS.length, FLAGS.R)


if __name__ == "__main__":
    flags.DEFINE_integer("length", 1048576, "Signal and FFT length.")
    flags.DEFINE_integer("R", 1024, "Downsampling factor.")

    app.run(main)
