from absl import app, flags, logging

from utils.plotter import serial_live_plotter

FLAGS = flags.FLAGS


def _parse_adc_data(read_data: str, num_sensors: int) -> tuple[float]:
    """Parses the ADC data serial output.

    Args:
        read_data: Read data from the serial port.
        num_sensors: Number of sensors to read.

    Returns:
        The ADC output in LSBs.
    """
    try:
        # The OpenMote prints (sequence number, channel, coarse code, medium
        # code, fine code, ADC output for each sensor, RSSI).
        adc_output = read_data.split(" ")[5:5 + num_sensors]
        return [float(data) for data in adc_output]
    except:
        logging.error("Failed to parse ADC data.")
        return [0] * num_sensorss


def main(argv):
    assert len(argv) == 1

    plotter = serial_live_plotter.SerialLivePlotter(
        FLAGS.port,
        FLAGS.baudrate,
        FLAGS.max_duration,
        lambda read_data: _parse_adc_data(read_data, FLAGS.num_sensors),
        num_traces=FLAGS.num_sensors,
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
    flags.DEFINE_integer("num_sensors",
                         4,
                         "Number of sensors to read.",
                         lower_bound=0)

    app.run(main)
