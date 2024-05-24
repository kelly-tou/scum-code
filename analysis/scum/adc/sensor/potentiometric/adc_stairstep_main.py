import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

from analysis.scum.adc.adc_config import ADC_CONFIGS, AdcConfig
from utils.regression.linear_regression import LinearRegression

FLAGS = flags.FLAGS


def plot_adc_stairstep(data: str, adc_config: AdcConfig) -> None:
    """Plots the ADC output over each sample.

    Args:
        data: Data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    concentration_column, adc_output_column = df.columns
    df_by_input = df.groupby(concentration_column)
    logging.info(df_by_input.describe())

    # Plot the stairstep line as the ADC output changes with each sample.
    fig, ax = plt.subplots(figsize=(12, 8))
    cutoff_indices = np.concatenate(
        ([0], np.where(np.diff(df[concentration_column]) != 0)[0] + 1,
         [len(df)]))
    for cutoff_index in range(1, len(cutoff_indices)):
        start_index, end_index = (
            cutoff_indices[cutoff_index - 1],
            cutoff_indices[cutoff_index],
        )
        df_concentration = df[start_index:end_index]
        (concentration,) = df_concentration[concentration_column].unique()
        data = df_concentration[adc_output_column]
        logging.info("%f M: mean = %f, stddev = %f", concentration, data.mean(),
                     data.std())
        data.plot.line(ax=ax)
    secax = ax.secondary_yaxis("right",
                               functions=(adc_config.lsb2volt,
                                          adc_config.volt2lsb))
    ax.set_title("ADC output over time")
    ax.set_xlabel("ADC sample")
    ax.set_ylabel("ADC output [LSB]")
    secax.set_ylabel("ADC output [V]")
    plt.show()


def plot_adc_means_stddevs(data: str, adc_config: AdcConfig) -> None:
    """Plots the ADC output means and standard deviations at each concentration.

    Args:
        data: Data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    concentration_column, adc_output_column = df.columns
    df_by_input = df.groupby(concentration_column)
    logging.info(df_by_input.describe())

    # Calculate the means and standard deviations of the ADC output for each concentration.
    means = df_by_input.mean()[adc_output_column]
    errors = df_by_input.std()[adc_output_column]

    # Perform a linear regression on the ADC output as a function of log10(concentration).
    linear_regression_data = means[means.index > 0.00001]
    linear_regression = LinearRegression(
        np.log10(linear_regression_data.index),
        adc_config.lsb2volt(linear_regression_data.values),
    )
    logging.info(
        "Linear regression in volts: m = %f, b = %f, residuals = %f",
        linear_regression.slope,
        linear_regression.y_intercept,
        linear_regression.residuals,
    )

    # Plot the means and standard deviations of the ADC output.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.errorbar(
        means.index,
        means.values,
        yerr=errors.values,
        label="ADC output [LSB]",
    )
    plt.plot(
        means.index,
        adc_config.volt2lsb(
            linear_regression.evaluate(np.log10(means.index.values))),
        "--",
        label="Linear regression",
    )
    ax.set_xscale("log")
    secax = ax.secondary_yaxis("right",
                               functions=(adc_config.lsb2volt,
                                          adc_config.volt2lsb))
    ax.set_title("Mean and standard deviation of the ADC output")
    ax.set_xlabel("Concentration [M]")
    ax.set_ylabel("ADC output [LSB]")
    secax.set_ylabel("ADC output [V]")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_adc_stairstep(FLAGS.data, ADC_CONFIGS[FLAGS.board])
    plot_adc_means_stddevs(FLAGS.data, ADC_CONFIGS[FLAGS.board])


if __name__ == "__main__":
    flags.DEFINE_string(
        "data",
        "analysis/scum/adc/sensor/potentiometric/data/adc_data_nitrate_vi_1.csv",
        "Data filename.",
    )
    flags.DEFINE_enum("board", "l35", ADC_CONFIGS.keys(), "SCuM board.")

    app.run(main)
