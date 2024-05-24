import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

from analysis.scum.adc.adc_config import ADC_CONFIGS, AdcConfig

FLAGS = flags.FLAGS


def plot_adc_sampling(data: str, adc_config: AdcConfig) -> None:
    """Plots the ADC output over time at different ADC sampling rates.

    Args:
        data: Data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    adc_sampling_rate_column, adc_output_column = df.columns
    df_by_input = df.groupby(adc_sampling_rate_column)
    logging.info(df_by_input.describe())

    # Plot the ADC output over time at different ADC sampling rates.
    fig, ax = plt.subplots(figsize=(12, 8))
    cutoff_indices = np.concatenate(
        ([0], np.where(np.diff(df[adc_sampling_rate_column]) != 0)[0] + 1,
         [len(df)]))
    for cutoff_index in range(1, len(cutoff_indices)):
        start_index, end_index = (
            cutoff_indices[cutoff_index - 1],
            cutoff_indices[cutoff_index],
        )
        df_adc_sampling_rate = df[start_index:end_index]
        (adc_sampling_rate,
        ) = df_adc_sampling_rate[adc_sampling_rate_column].unique()
        data = df_adc_sampling_rate[adc_output_column]
        data.reset_index()[adc_output_column].plot.line(ax=ax)
    ax.set_title("ADC output over time")
    ax.set_xlabel("ADC sample")
    ax.set_ylabel("ADC output [LSB]")
    plt.show()


def plot_adc_sampling_means_stddevs(data: str, adc_config: AdcConfig) -> None:
    """Plots the ADC output means and standard deviations at different ADC sampling rates.

    Args:
        data: Data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    adc_sampling_rate_column, adc_output_column = df.columns
    df_by_input = df.groupby(adc_sampling_rate_column)
    logging.info(df_by_input.describe())

    # Calculate the means and standard deviations of the ADC output for each ADC sampling rate.
    means = df_by_input.mean()[adc_output_column]
    errors = df_by_input.std()[adc_output_column]

    # Plot the ADC output means and standard deviations at different ADC sampling rates.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.errorbar(
        means.index,
        means.values,
        yerr=errors.values,
        label="ADC output [LSB]",
    )
    ax.set_xscale("log")
    ax.set_title("Mean and standard deviation of the ADC output")
    ax.set_xlabel("Loop cycles between ADC samples")
    ax.set_ylabel("ADC output [LSB]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_adc_sampling(FLAGS.data, ADC_CONFIGS[FLAGS.board])
    plot_adc_sampling_means_stddevs(FLAGS.data, ADC_CONFIGS[FLAGS.board])


if __name__ == "__main__":
    flags.DEFINE_string(
        "data",
        "analysis/scum/adc/sensor/potentiometric/data/adc_data_nitrate_vi_1_sampling.csv",
        "Data filename.",
    )
    flags.DEFINE_enum("board", "l35", ADC_CONFIGS.keys(), "SCuM board.")

    app.run(main)
