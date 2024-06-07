import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from absl import app, flags, logging

from utils.visualization.color_maps import COLOR_MAPS

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
    logging.info(df.describe())

    # Plot the standard error as a function of the grid dimensions.
    plt.style.use("science")
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": "3d"},
    )
    surf = ax.plot_trisurf(
        df[num_rows_column],
        df[num_cols_column],
        np.sqrt(df[standard_error_squared_column]),
        cmap=COLOR_MAPS["parula"],
        antialiased=False,
    )
    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Number of columns")
    ax.set_zlabel("Standard error")
    ax.view_init(30, -45)
    plt.colorbar(surf)
    plt.show()

    # Plot the squared standard error as a function of the grid dimensions.
    plt.style.use("science")
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": "3d"},
    )
    surf = ax.plot_trisurf(
        df[num_rows_column],
        df[num_cols_column],
        df[standard_error_squared_column],
        cmap=COLOR_MAPS["parula"],
        antialiased=False,
    )
    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Number of columns")
    ax.set_zlabel("Squared standard error")
    ax.view_init(30, -45)
    plt.colorbar(surf)
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_standard_error_sweep(FLAGS.standard_errors)


if __name__ == "__main__":
    flags.DEFINE_string(
        "standard_errors",
        "simulation/differential_mesh/data/grid_standard_errors.csv",
        "Standard errors.")

    app.run(main)
