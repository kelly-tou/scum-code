import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from absl import app, flags, logging

from analysis.scum.adc.adc_config import ADC_CONFIGS, AdcConfig
from analysis.scum.adc.enob.adc_data import AdcData
from utils.regression.linear_regression import LinearRegression
from utils.regression.parabolic_regression import ParabolicRegression

FLAGS = flags.FLAGS


def plot_adc_samples(data: str, adc_config: AdcConfig) -> None:
    """Plots the ADC samples over time.

    Args:
        data: Data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    adc_output_column, = df.columns
    logging.info(df.describe())
    adc_data = AdcData(df[adc_output_column], adc_config.max_sampling_rate)
    adc_data.disambiguate_msb_9()

    # Plot the ADC samples.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(adc_data.time_axis, adc_data.samples)
    ax.set_title("ADC samples over time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ADC output [LSB]")
    plt.show()

    # Plot the difference between consecutive ADC samples.
    adc_sample_differences = np.diff(adc_data.samples)
    logging.info(
        "ADC sample differences: mean = %f, stddev = %f",
        np.mean(adc_sample_differences),
        np.std(adc_sample_differences),
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(adc_data.time_axis[1:], np.diff(adc_data.samples))
    ax.set_title("Difference of consecutive ADC samples over time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ADC output difference [LSB]")
    plt.show()

    # Plot the FFT of the ADC samples.
    adc_sample_fft = np.fft.fft(adc_data.samples)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(np.abs(np.fft.fft(adc_data.samples)))
    ax.set_title("FFT of the ADC samples")
    ax.set_xlabel("FFT bin")
    ax.set_ylabel("FFT output")
    plt.show()

    # There should be three peaks in the FFT spectrum, one for the DC offset
    # and two corresponding to the sinusoid's frequency.
    # Plot the ADC samples with the signal.
    top_three_peak_indices = np.argpartition(np.abs(adc_sample_fft), -3)[-3:]
    logging.info("Top three peak indices: %s", str(top_three_peak_indices))

    def construct_signal(frequency: float, phase: float):
        """Constructs the sinnusoid with the given frequency and phase.

        Args:
            frequency: Signal frequency.
            phase: Signal phase.

        Returns:
            Sinusoid corresponding to the given frequency and phase.
        """
        return (adc_config.amplitude_lsbs *
                np.cos(2 * np.pi * frequency * adc_data.time_axis + phase) +
                adc_config.offset_lsbs)

    # Use the inverse FFT to find the sinusoid.
    signal_fft = np.zeros(adc_sample_fft.shape, dtype=adc_sample_fft.dtype)
    signal_fft[top_three_peak_indices] = adc_sample_fft[top_three_peak_indices]
    signal_ifft = np.fft.ifft(signal_fft)

    # Use the FFT to find the frequency and phase of the sinusoid.
    signal_fft_bin = np.sort(top_three_peak_indices)[1]
    # Parabolically interpolate the signal frequency.
    signal_fft_parabola_indices = np.array([-1, 0, 1]) + signal_fft_bin
    # TODO(titan): Handle wraparound.
    signal_fft_magnitude_parabola = ParabolicRegression(
        signal_fft_parabola_indices,
        np.abs(adc_sample_fft[signal_fft_parabola_indices]))
    signal_fft_interpolated_bin = signal_fft_magnitude_parabola.peak()[0]
    signal_frequency = (signal_fft_interpolated_bin *
                        adc_config.max_sampling_rate / adc_data.num_samples)
    # Linearly interpolate the signal phase.
    signal_fft_neighboring_bins = np.array([
        int(np.floor(signal_fft_interpolated_bin)),
        int(np.ceil(signal_fft_interpolated_bin)),
    ])
    signal_fft_phase_line = LinearRegression(
        signal_fft_neighboring_bins,
        np.angle(adc_sample_fft)[signal_fft_neighboring_bins],
    )
    signal_phase = signal_fft_phase_line.evaluate(signal_fft_interpolated_bin)
    logging.info("Signal frequency = %f, phase = %f", signal_frequency,
                 signal_phase)
    signal_fixed_amplitude = construct_signal(signal_frequency, signal_phase)

    # Use an optimizer to find the optimal frequency and phase of the sinusoid.
    def cost(x: np.ndarray):
        """Calculates how well the given signal frequency and phase correlate with the ADC data.

        Args:
            x: Two-dimensional vector consisting of the signal frequency and phase.

        Returns:
            The negative correlation between the given signal and the ADC data.
        """
        return -np.inner(adc_data.samples, construct_signal(*x))

    optimization_results = scipy.optimize.minimize(
        cost, np.array([signal_frequency, signal_phase]), method="Nelder-Mead")
    if not optimization_results.success:
        logging.warning("Optimization failed with message: %s",
                        optimization_results.message)
    logging.info("Optimized signal frequency = %f, phase = %f",
                 *optimization_results.x)
    signal_optimized = construct_signal(*optimization_results.x)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(adc_data.time_axis, adc_data.samples, label="ADC samples")
    plt.plot(
        adc_data.time_axis,
        np.abs(signal_ifft),
        label="Reconstructed sinusoid with inverse FFT",
    )
    plt.plot(
        adc_data.time_axis,
        signal_fixed_amplitude,
        label=
        "Reconstructed sinusoid with fixed amplitude and frequency and phase interpolation",
    )
    plt.plot(
        adc_data.time_axis,
        signal_optimized,
        label="Reconstructed sinusoid with fixed amplitude after optimization",
    )
    ax.set_title("ADC samples over time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ADC output [LSB]")
    plt.legend()
    plt.show()

    # Calculate the noise.
    noise = adc_data.samples - signal_optimized
    logging.info("Noise: mean = %f, standard deviation = %f", np.mean(noise),
                 np.std(noise))
    noise_rms = np.sqrt(np.mean((noise - np.mean(noise))**2))
    enob = np.log2(2**9 / np.sqrt(12) / noise_rms)
    logging.info("Noise = %f LSB, ENOB = %f bits", noise_rms, enob)


def main(argv):
    assert len(argv) == 1
    plot_adc_samples(FLAGS.data, ADC_CONFIGS[FLAGS.board])


if __name__ == "__main__":
    flags.DEFINE_string("data",
                        "analysis/scum/adc/enob/data/adc_data_100hz.csv",
                        "Data filename.")
    flags.DEFINE_enum("board", "l35", ADC_CONFIGS.keys(), "SCuM board.")

    app.run(main)
