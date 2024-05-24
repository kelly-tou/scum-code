import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

from analysis.scum.adc.adc_config import ADC_CONFIGS, AdcConfig
from analysis.scum.adc.enob.adc_data import AdcData

FLAGS = flags.FLAGS


def plot_adc_samples_with_scope_and_averaging(data: str, scope_data: str,
                                              adc_config: AdcConfig) -> None:
    """Plots the ADC samples over time after averaging them.

    Args:
        data: Data filename.
        scope_data: Oscilloscope data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    adc_output_column, = df.columns
    logging.info(df.describe())
    adc_data = AdcData(df[adc_output_column], adc_config.max_sampling_rate)
    adc_data.disambiguate_msb_9()

    # Open the ADC scope data file.
    scope_df = pd.read_csv(scope_data, comment="#")
    time_column, gpio_column, sinusoid_column = scope_df.columns

    # Plot the time difference between ADC samples.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(np.diff(scope_df[time_column]))
    ax.set_xlabel("ADC sample")
    ax.set_ylabel("Time difference [s]")
    ax.set_title("Time difference between ADC samples")
    plt.show()

    # Plot the ADC samples and the reference sinusoid.
    reference_sinusoid_lsbs = (
        adc_config.amplitude_lsbs / adc_config.amplitude_volts *
        (scope_df[sinusoid_column] - adc_config.offset_volts) +
        adc_config.offset_lsbs)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(scope_df[time_column], adc_data.samples, label="ADC samples")
    plt.plot(scope_df[time_column],
             reference_sinusoid_lsbs,
             label="Reference sinusoid")
    ax.set_title("ADC samples over time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ADC output [LSB]")
    plt.legend()
    plt.show()

    # Calculate the noise.
    noise = adc_data.samples - reference_sinusoid_lsbs
    logging.info("Noise: mean = %f, standard deviation = %f", np.mean(noise),
                 np.std(noise))
    noise_rms = np.sqrt(np.mean((noise - np.mean(noise))**2))
    enob = np.log2(2**9 / np.sqrt(12) / noise_rms)
    logging.info("Noise = %f LSB, ENOB = %f bits", noise_rms, enob)


def main(argv):
    assert len(argv) == 1
    plot_adc_samples_with_scope_and_averaging(FLAGS.data, FLAGS.scope_data,
                                              ADC_CONFIGS[FLAGS.board])


if __name__ == "__main__":
    flags.DEFINE_string(
        "data",
        "analysis/scum/adc/enob/data/adc_data_with_scope_1hz.csv",
        "Data filename.",
    )
    flags.DEFINE_string(
        "scope_data",
        "analysis/scum/adc/enob/data/scope_1hz.csv",
        "Oscilloscope data filename.",
    )
    flags.DEFINE_enum("board", "l35", ADC_CONFIGS.keys(), "SCuM board.")

    app.run(main)
