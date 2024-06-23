import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from absl import app, flags

from simulation.cic_filter.cic_filter_decimator import CicFilterDecimator

FLAGS = flags.FLAGS

# Comb filter.
COMB_FILTER = np.array([0.25, 0.75, 0.75, 0.25])


def plot_cic_filter_output(signal: np.ndarray, R: int, N: int) -> None:
    """Plots the output of a standard CIC filter.

    Args:
        signal: Signal.
        R: Downsampling factor.
        N: Number of stages.
    """
    cic_filter_decimator = CicFilterDecimator(R, N)
    response = cic_filter_decimator.filter(signal)

    # Plot the signal.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.stem(response)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Output")
    plt.show()

    fft_length = len(signal)
    omega, response_fft_abs = cic_filter_decimator.calculate_spectrum_magnitude(
        response, fft_length)

    # Plot the spectrum.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(omega, 20 * np.log10(response_fft_abs))
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("FFT magnitude [dB]")
    plt.show()


def plot_cic_filter_comb_filter(signal: np.ndarray, R: int, N: int) -> None:
    """Plots the output of a CIC filter with a custom comb filter.

    Args:
        signal: Signal.
        R: Downsampling factor.
        N: Number of stages.
    """
    # Simulate the CIC filter with a custom comb filter.
    cic_filter_decimator = CicFilterDecimator(R, N, COMB_FILTER)
    response = cic_filter_decimator.filter(signal)

    # Simulate the standard CIC filter.
    cic_filter_decimator_standard = CicFilterDecimator(R, N)
    response_standard = cic_filter_decimator_standard.filter(signal)

    fft_length = len(signal)
    omega, response_fft_abs = (
        cic_filter_decimator.calculate_spectrum_magnitude(response, fft_length))
    omega_standard, response_standard_fft_abs = (
        cic_filter_decimator_standard.calculate_spectrum_magnitude(
            response_standard, fft_length))

    # Plot the signal.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.stem(response)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Output")
    plt.show()

    # Plot the spectrum.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(omega,
            20 * np.log10(response_fft_abs / np.max(response_fft_abs)),
            label=f"Comb filter: {cic_filter_decimator.comb_filter}")
    ax.plot(
        omega_standard,
        20 *
        np.log10(response_standard_fft_abs / np.max(response_standard_fft_abs)),
        "--",
        label=(f"Standard comb filter: "
               f"{cic_filter_decimator_standard.comb_filter}"))
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("Normalized FFT magnitude [dB]")
    ax.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    signal_length = int(FLAGS.fs * FLAGS.T)
    if FLAGS.fin is None:
        signal = np.zeros(signal_length)
        signal[0] = 1
    else:
        signal = np.cos(2 * np.pi * FLAGS.fin / FLAGS.fs *
                        np.arange(signal_length))
    if FLAGS.noise is not None:
        signal += np.random.normal(scale=FLAGS.noise, size=signal.shape)

    plot_cic_filter_output(signal, FLAGS.R, FLAGS.N)
    plot_cic_filter_comb_filter(signal, FLAGS.R, FLAGS.N)


if __name__ == "__main__":
    flags.DEFINE_float("fs", 1000000, "Sampling frequency.")
    flags.DEFINE_float("T", 1, "Maximum time to simulate.")
    flags.DEFINE_integer("R", 1024, "Downsampling factor.")
    flags.DEFINE_integer("N", 1, "Number of stages.")
    flags.DEFINE_float("fin", None, "Input frequency.")
    flags.DEFINE_float("noise", None, "Noise amplitude.")

    app.run(main)
