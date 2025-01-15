import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from absl import app, flags, logging
from matplotlib.ticker import FuncFormatter

from analysis.scum.radio.tuning.tuning_code import TuningCode

FLAGS = flags.FLAGS

# Tuning code column.
TUNING_CODE_COLUMN = "Tuning code"

# TX/RX label.
TX_LABEL = "TX"
RX_LABEL = "RX"


def _tuning_code_formatter(x: float, position: float) -> str:
    """Formats the tuning code tick labels.

    Args:
        x: Tick value.
        position: Position.

    Returns:
        The string containing the coarse, mid, and fine codes.
    """
    tuning_code = TuningCode.tuning_code_to_coarse_mid_fine(int(np.abs(x)))
    if x < 0:
        return rf"$-{tuning_code}$"
    return rf"${tuning_code}$"


def plot_tuning_codes(data: str) -> None:
    """Plots the TX and RX tuning codes by 802.15.4 channel.

    Args:
        data: Data filename.
    """
    # Open the tuning codes file.
    df = pd.read_csv(data, comment="#")
    (
        tx_rx_column,
        channel_column,
        coarse_column,
        mid_column,
        fine_column,
    ) = df.columns
    df[TUNING_CODE_COLUMN] = TuningCode.coarse_mid_fine_to_tuning_code(
        df[coarse_column], df[mid_column], df[fine_column])
    logging.info(df.describe())

    # Plot the TX and RX tuning codes.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, (tx_rx, tx_rx_group) in enumerate(df.groupby(tx_rx_column)):
        tx_rx_group.plot.scatter(
            TUNING_CODE_COLUMN,
            channel_column,
            ax=ax,
            c=f"C{index}",
            alpha=0.5,
            label=tx_rx,
            legend=True,
        )
    ax.set_title("TX and RX tuning codes")
    ax.xaxis.set_major_formatter(FuncFormatter(_tuning_code_formatter))
    plt.show()

    # Plot the difference between the TX and RX tuning codes.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 6))
    for (channel, channel_group) in df.groupby(channel_column):
        tx = channel_group.groupby(tx_rx_column).get_group(TX_LABEL)
        rx = channel_group.groupby(tx_rx_column).get_group(RX_LABEL)
        tx_rx_differences = (tx[TUNING_CODE_COLUMN].to_numpy()[:, None] -
                             rx[TUNING_CODE_COLUMN].to_numpy()).flatten()
        plt.scatter(
            np.full(tx_rx_differences.shape, channel),
            tx_rx_differences,
        )
    ax.set_title("Difference between TX and RX tuning codes")
    ax.set_xlabel("Channel")
    ax.set_ylabel("TX - RX difference")
    ax.yaxis.set_major_formatter(FuncFormatter(_tuning_code_formatter))
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_tuning_codes(FLAGS.data)


if __name__ == "__main__":
    flags.DEFINE_string("data",
                        "analysis/scum/radio/tuning/data/tuning_codes_1.csv",
                        "Data filename.")

    app.run(main)
