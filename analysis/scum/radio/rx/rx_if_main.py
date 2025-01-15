import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import scipy.stats
from absl import app, flags, logging

FLAGS = flags.FLAGS


def plot_rx_if_data(data: str) -> None:
    """Plots the IF estimates from the received packets.

    Args:
        data: Data filename.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    (if_estimates_column,) = df.columns
    logging.info(df.describe())

    # Plot the IF estimates over time.
    if_estimates = df[if_estimates_column]
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 6))
    if_estimates.plot(ax=ax)
    ax.set_title("IF estimates of received 802.15.4 packets")
    ax.set_xlabel("Packet index")
    ax.set_ylabel("IF estimate [counts]")
    plt.show()

    # Plot a histogram of the IF estimates.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 6))
    minimum_if_estimate = if_estimates.min()
    maximum_if_estimate = if_estimates.max()
    bins = np.arange(minimum_if_estimate - 0.5, maximum_if_estimate + 1)
    if_estimates.hist(ax=ax, bins=bins)

    # Plot the Gaussian fit.
    secax = ax.twinx()
    gaussian_fit = scipy.stats.norm.pdf(
        bins,
        if_estimates.mean(),
        if_estimates.std(),
    )
    secax.plot(bins, gaussian_fit, "r")
    secax.set_ylim(bottom=0)
    ax.set_title("IF estimates of received 802.15.4 packets")
    ax.set_xlabel("IF estimate [counts]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_rx_if_data(FLAGS.data)


if __name__ == "__main__":
    flags.DEFINE_string("data", "analysis/scum/radio/rx/data/rx_if_data.csv",
                        "Data filename.")

    app.run(main)
