import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from absl import app, flags, logging

from utils.regression.logarithmic_regression import LogarithmicRegression

FLAGS = flags.FLAGS


def plot_standard_error_sweep(standard_errors: str) -> None:
    """Plots the standard error of the node potentials as a function of the
    grid dimensions.

    Args:
        standard_errors: Standard errors filename.
    """
    # Open the standard errors file.
    df = pd.read_csv(standard_errors, comment="#")
    num_rows_column, num_cols_column, standard_error_squared_column = df.columns
    assert df[num_rows_column].equals(df[num_cols_column])
    logging.info(df.describe())

    # Plot the standard error as a function of the grid dimensions.
    plt.style.use("science")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df[num_rows_column], np.sqrt(df[standard_error_squared_column]))
    ax.set_xlabel("Square grid dimensions")
    ax.set_ylabel("Standard error")
    plt.show()

    # Perform a logarithmic regression on the squared standard error.
    logarithmic_regression = LogarithmicRegression(
        df[num_rows_column], df[standard_error_squared_column])
    logging.info("Logarithmic regression: a = %f, b = %f",
                 logarithmic_regression.a, logarithmic_regression.b)

    # Plot the squared standard error as a function of the grid dimensions.
    plt.style.use("science")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df[num_rows_column], df[standard_error_squared_column])
    ax.plot(df[num_rows_column],
            logarithmic_regression.evaluate(df[num_rows_column]), "--")
    ax.set_xlabel("Square grid dimensions")
    ax.set_ylabel("Squared standard error")
    plt.show()

    # Plot the squared standard error as a function of the grid dimensions on a
    # semilog x-axis.
    plt.style.use("science")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.semilogx(df[num_rows_column],
                df[standard_error_squared_column],
                linewidth=2)
    ax.semilogx(df[num_rows_column],
                logarithmic_regression.evaluate(df[num_rows_column]), "--")
    ax.set_xlabel("Square grid dimensions")
    ax.set_ylabel("Squared standard error")
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_standard_error_sweep(FLAGS.standard_errors)


if __name__ == "__main__":
    flags.DEFINE_string(
        "standard_errors",
        "simulation/differential_mesh/data/square_standard_errors.csv",
        "Standard errors.")

    app.run(main)
