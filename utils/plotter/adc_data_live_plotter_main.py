from absl import app, flags, logging

from utils.plotter import serial_live_plotter

FLAGS = flags.FLAGS


def _parse_adc_data(read_data: str) -> float:
    """Parses the ADC data serial output.

    Args:
        read_data: Read data from the serial port.

    Returns:
        The ADC output in LSBs.
    """
    try:
        # The OpenMote prints (sequence number, channel, coarse code, medium
        # code, fine code, ADC output, RSSI).
        adc_output = read_data.split(" ")[5]
        return float(adc_output)
    except:
        logging.error("Failed to parse ADC data.")
        return 0


def main(argv):
    assert len(argv) == 1

    plotter = serial_live_plotter.SerialLivePlotter(FLAGS.port,
                                                    FLAGS.baudrate,
                                                    FLAGS.max_duration,
                                                    _parse_adc_data,
                                                    title="ADC data",
                                                    xlabel="Time [s]",
                                                    ylabel="ADC output [LSB]",
                                                    ymin=0,
                                                    ymax=512)
    plotter.run()


if __name__ == "__main__":
    flags.DEFINE_string("port", "/dev/tty.usbserial-A10M1IFE",
                        "Serial port to plot for.")
    flags.DEFINE_integer("baudrate",
                         19200,
                         "Serial port baud rate.",
                         lower_bound=0)
    flags.DEFINE_integer("max_duration",
                         30,
                         "Maximum duration to plot in seconds.",
                         lower_bound=0)

    app.run(main)
