import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

from analysis.scum.adc.adc_config import ADC_CONFIGS, AdcConfig
from utils.regression.linear_regression import LinearRegression

FLAGS = flags.FLAGS

# Number of ADC bits.
NUM_ADC_BITS = 10

# Maximum sensor input voltage in V.
SENSOR_INPUT_MAX = 0.5  # V


def plot_adc_characterization(data: str, adc_config: AdcConfig) -> None:
    """Plots the ADC output over the ADC input signal.

    Args:
        data: Data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC characterization data file.
    df = pd.read_csv(data, comment="#")
    voltage_column, adc_output_column = df.columns
    df_by_input = df.groupby(voltage_column)
    logging.info(df_by_input.describe())

    # Calculate the means and standard deviations of the ADC output for each input.
    means = df_by_input.mean()[adc_output_column]
    errors = df_by_input.std()[adc_output_column]

    # Plot the means compared to an ideal ADC.
    input_range = np.linspace(min(df_by_input.groups.keys()),
                              max(df_by_input.groups.keys()), 100)
    fig, ax = plt.subplots(figsize=(12, 8))
    jump_indices = np.concatenate(
        ([0], np.where(np.diff(means.values) < 0)[0] + 1, [len(means)]))
    for jump_index in range(1, len(jump_indices)):
        error_bar = plt.errorbar(
            means.index[jump_indices[jump_index - 1]:jump_indices[jump_index]],
            means.values[jump_indices[jump_index - 1]:jump_indices[jump_index]],
            yerr=errors.values[jump_indices[jump_index -
                                            1]:jump_indices[jump_index]],
            color="C0",
            label="ADC output [LSB]",
        )
    (ideal_line,) = plt.plot(
        input_range,
        input_range / adc_config.ldo_output * (2**NUM_ADC_BITS - 1),
        "--",
        color="C1",
        label="Ideal ADC output [LSB]",
    )
    ax.set_title("Mean of the ADC output vs. ideal ADC output")
    ax.set_xlabel("Input voltage [V]")
    ax.set_ylabel("ADC output [LSB]")
    plt.legend(handles=[error_bar, ideal_line])
    plt.show()

    # Plot the means and standard deviations of the ADC output for each input.
    fig, ax = plt.subplots(figsize=(12, 8))
    means.plot.bar(yerr=errors, ax=ax).legend(loc="upper left")
    ax.bar_label(
        ax.containers[1],
        labels=np.round(means.values, decimals=2),
        label_type="center",
    )
    ax.bar_label(
        ax.containers[1],
        labels=np.round(errors.values, decimals=2),
        label_type="edge",
    )
    ax.set_title("Mean and standard deviation of the ADC output")
    ax.set_xlabel("Input voltage [V]")
    ax.set_ylabel("ADC output [LSB]")
    plt.show()


def plot_adc_characterization_sensor(data: str, adc_config: AdcConfig) -> None:
    """Plots the ADC output over the ADC input signal for the voltage range of the sensor.

    Args:
        data: Data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC characterization data file.
    df = pd.read_csv(data, comment="#")
    voltage_column, adc_output_column = df.columns
    df_by_input = df[df[voltage_column] <= SENSOR_INPUT_MAX].groupby(
        voltage_column)
    logging.info(df_by_input.describe())

    # Calculate the means and standard deviations of the ADC output for each input.
    means = df_by_input.mean()[adc_output_column]
    errors = df_by_input.std()[adc_output_column]

    # Perform a linear regression on the ADC output as a function of the input.
    linear_regression = LinearRegression(list(df_by_input.groups.keys()), means)
    logging.info("Ideal ADC: m = %f, b = 0",
                 1 / adc_config.ldo_output * (2**NUM_ADC_BITS - 1))
    logging.info(
        "SCuM ADC: m = %f, b = %f, residuals = %f",
        linear_regression.slope,
        linear_regression.y_intercept,
        linear_regression.residuals,
    )

    # Plot the means with a linear regression and an ideal ADC.
    input_range = np.linspace(min(df_by_input.groups.keys()),
                              max(df_by_input.groups.keys()), 100)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.errorbar(
        means.index,
        means.values,
        yerr=errors.values,
        label="ADC output [LSB]",
    )
    plt.plot(
        input_range,
        input_range / adc_config.ldo_output * (2**NUM_ADC_BITS - 1),
        "--",
        label="Ideal ADC output [LSB]",
    )
    plt.plot(
        input_range,
        linear_regression.evaluate(input_range),
        "--",
        label=f"Linear regression of the ADC output [LSB]",
    )
    ax.set_title("Mean of the ADC output vs. ideal ADC output")
    ax.set_xlabel("Input voltage [V]")
    ax.set_ylabel("ADC output [LSB]")
    plt.legend()
    plt.show()

    # Plot the means and standard deviations of the ADC output for each input.
    fig, ax = plt.subplots(figsize=(12, 8))
    means.plot.bar(yerr=errors, ax=ax).legend(loc="upper left")
    ax.bar_label(
        ax.containers[1],
        labels=np.round(means.values, decimals=2),
        label_type="center",
    )
    ax.bar_label(
        ax.containers[1],
        labels=np.round(errors.values, decimals=2),
        label_type="edge",
    )
    ax.set_title("Mean and standard deviation of the ADC output")
    ax.set_xlabel("Input voltage [V]")
    ax.set_ylabel("ADC output [LSB]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_adc_characterization(FLAGS.data, ADC_CONFIGS[FLAGS.board])
    plot_adc_characterization_sensor(FLAGS.data, ADC_CONFIGS[FLAGS.board])


if __name__ == "__main__":
    flags.DEFINE_string(
        "data", "analysis/scum/adc/data/adc_characterization_data_l35.csv",
        "Data filename.")
    flags.DEFINE_enum("board", "l35", ADC_CONFIGS.keys(), "SCuM board.")

    app.run(main)
