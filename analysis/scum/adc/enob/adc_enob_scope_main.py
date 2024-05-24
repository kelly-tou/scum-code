import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

from analysis.scum.adc.adc_config import ADC_CONFIGS, AdcConfig
from analysis.scum.adc.enob.adc_data import AdcData

FLAGS = flags.FLAGS

# Number of ADC samples to average over.
NUM_SAMPLES_TO_AVERAGE = np.arange(1, 1001)
NUM_SAMPLES_TO_AVERAGE_AND_PLOT = [1, 10, 100, 1000]


def plot_adc_samples_with_scope(data: str, scope_data: str,
                                adc_config: AdcConfig) -> None:
    """Plots the ADC samples over time.

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

    # Calculate the noise after averaging over ADC samples.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(scope_df[time_column],
             reference_sinusoid_lsbs,
             label="Reference sinusoid")
    averaging_enob = []
    for num_samples_to_average in NUM_SAMPLES_TO_AVERAGE:
        averaging_filter = np.ones(
            num_samples_to_average) / num_samples_to_average
        adc_samples = np.convolve(
            adc_data.samples, averaging_filter,
            mode="same")[num_samples_to_average:-num_samples_to_average]
        noise = (adc_samples - reference_sinusoid_lsbs[
            num_samples_to_average:-num_samples_to_average])
        logging.info(
            "Averaging over %d samples: noise: mean = %f, standard deviation = %f",
            num_samples_to_average,
            np.mean(noise),
            np.std(noise),
        )
        noise_rms = np.sqrt(np.mean((noise - np.mean(noise))**2))
        enob = np.log2(2**9 / np.sqrt(12) / noise_rms)
        averaging_enob.append(enob)
        logging.info(
            "Averaging over %d samples: noise = %f LSB, ENOB = %f bits",
            num_samples_to_average,
            noise_rms,
            enob,
        )

        # Plot the averaged ADC samples.
        if num_samples_to_average in NUM_SAMPLES_TO_AVERAGE_AND_PLOT:
            plt.plot(
                scope_df[time_column]
                [num_samples_to_average:-num_samples_to_average],
                adc_samples,
                label=f"Averaging over {num_samples_to_average} samples",
            )
    ax.set_title("ADC samples over time after averaging")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ADC output [LSB]")
    plt.legend()
    plt.show()

    # Plot the ENOB as a function of the number of ADC samples to average over.
    logging.info(
        "Maximum ENOB achieved at averaging over %d samples: max ENOB = %f bits",
        NUM_SAMPLES_TO_AVERAGE[np.argmax(averaging_enob)],
        np.max(averaging_enob),
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(NUM_SAMPLES_TO_AVERAGE, averaging_enob)
    ax.set_title(f"ENOB after averaging over ADC samples")
    ax.set_xlabel("Number of ADC samples to average over")
    ax.set_ylabel("ENOB [bits]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_adc_samples_with_scope(FLAGS.data, FLAGS.scope_data,
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
