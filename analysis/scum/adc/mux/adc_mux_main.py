import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from absl import app, flags, logging

from analysis.scum.adc.adc_config import ADC_CONFIGS, AdcConfig

FLAGS = flags.FLAGS


def plot_muxed_adc_data(data: str, adc_config: AdcConfig) -> None:
    """Plots the muxed ADC samples.

    Each column in the data corresponds to a sensor.

    Args:
        data: Data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    logging.info(df.describe())

    # Plot the muxed ADC samples.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    df.plot(ax=ax)
    secax = ax.secondary_yaxis("right",
                               functions=(adc_config.lsb2volt,
                                          adc_config.volt2lsb))
    ax.set_title("Muxed ADC output")
    ax.set_xlabel("ADC sample")
    ax.set_ylabel("ADC output [LSB]")
    secax.set_ylabel("ADC output [V]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_muxed_adc_data(FLAGS.data, ADC_CONFIGS[FLAGS.board])


if __name__ == "__main__":
    flags.DEFINE_string("data", "analysis/scum/adc/mux/data/adc_mux_data_3.csv",
                        "Data filename.")
    flags.DEFINE_enum("board", "m2", ADC_CONFIGS.keys(), "SCuM board.")

    app.run(main)
