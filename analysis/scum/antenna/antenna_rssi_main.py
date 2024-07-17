import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from absl import app, flags, logging

FLAGS = flags.FLAGS


def plot_antenna_rssi(data: str) -> None:
    """Plots the antenna RSSI.

    Each column in the data corresponds to an antenna.

    Args:
        data: Data filename.
    """
    # Open the antenna RSSI data file.
    df = pd.read_csv(data, comment="#")
    logging.info(df.describe())
    distance_column = df.columns[0]

    # Group by antenna.
    df_by_antenna = df.groupby(distance_column)

    # Plot the mean and standard deviation of the RSSI.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    df_by_antenna.mean().plot(ax=ax, yerr=df_by_antenna.std(), marker="^")
    ax.set_ylabel("RSSI [dBm]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_antenna_rssi(FLAGS.data)


if __name__ == "__main__":
    flags.DEFINE_string(
        "data", "analysis/scum/antenna/data/antenna_rssi_data_field.csv",
        "Data filename.")

    app.run(main)
