import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

from utils.regression.linear_regression import LinearRegression

FLAGS = flags.FLAGS


def plot_capacitor_voltage(data: str, data_without_scum: str) -> None:
    """Plots the capacitor voltage over time.

    Args:
        data: Data filename.
        data_without_scum: Data filename without SCuM.
    """
    # Open the scope data file.
    df = pd.read_csv(data)
    time_column, capacitor_column, gpio_column = df.columns
    logging.info(df.describe())

    # Open the scope data file without SCuM.
    df_without_scum = pd.read_csv(data_without_scum)
    time_column_without_scum, capacitor_column_without_scum = df_without_scum.columns

    # Perform a linear regression on the capacitor voltage.
    capacitor_voltage_line = LinearRegression(df[time_column],
                                              df[capacitor_column])
    capacitor_voltage_line_without_scum = LinearRegression(
        df_without_scum[time_column_without_scum],
        df_without_scum[capacitor_column_without_scum],
    )
    logging.info("Capacitor voltage: m = %f", capacitor_voltage_line.slope)
    logging.info("Capacitor voltage without SCuM: m = %f",
                 capacitor_voltage_line_without_scum.slope)

    # Plot the capacitor voltage over time.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(df[time_column],
             df[capacitor_column],
             label="Capacitor voltage with SCuM ADC")
    plt.plot(
        df[time_column],
        capacitor_voltage_line.slope * df[time_column] +
        capacitor_voltage_line.y_intercept,
        "--",
    )
    plt.plot(
        df_without_scum[time_column_without_scum],
        df_without_scum[capacitor_column_without_scum],
        label="Capacitor voltage without SCuM ADC",
    )
    plt.plot(
        df_without_scum[time_column_without_scum],
        capacitor_voltage_line_without_scum.slope *
        df_without_scum[time_column_without_scum] +
        capacitor_voltage_line_without_scum.y_intercept,
        "--",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Capacitor voltage [V]")
    ax.set_title("Capacitor voltage over time")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_capacitor_voltage(FLAGS.data, FLAGS.data_without_scum)


if __name__ == "__main__":
    flags.DEFINE_string("data",
                        "analysis/scum/adc/capacitor/data/scope_1uf.csv",
                        "Data filename.")
    flags.DEFINE_string(
        "data_without_scum",
        "analysis/scum/adc/capacitor/data/scope_1uf_without_scum.csv",
        "Data filename.",
    )

    app.run(main)
