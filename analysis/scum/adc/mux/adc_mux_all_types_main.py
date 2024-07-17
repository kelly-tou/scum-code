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

    # Separate the ADC readouts and the time constants.
    adc_output_columns = df.columns[:2]
    time_constant_columns = df.columns[2:]

    # Convert from ADC LSBs to volts.
    df[adc_output_columns] = df[adc_output_columns].apply(adc_config.lsb2volt)

    # Evaluate the time constant ratios.
    for column in time_constant_columns:
        df[column] = df[column].apply(pd.eval)

    # Plot the muxed ADC readouts and time constants.
    plt.style.use(["science", "grid"])
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    df[adc_output_columns].plot(ax=ax1, color="C0", style=["-.", ":"])
    df[time_constant_columns].plot(ax=ax2, color="C3", style=["--", "-"])
    ax1.set_title("Muxed sensor readouts")
    ax1.set_xlabel("Readout index")
    ax1.set_ylabel("ADC output [V]")
    ax2.set_ylabel("Time constant [s]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_muxed_adc_data(FLAGS.data, ADC_CONFIGS[FLAGS.board])


if __name__ == "__main__":
    flags.DEFINE_string(
        "data", "analysis/scum/adc/mux/data/adc_mux_data_all_types_1.csv",
        "Data filename.")
    flags.DEFINE_enum("board", "m2", ADC_CONFIGS.keys(), "SCuM board.")

    app.run(main)
