from absl import app, flags

from utils.network.s1p_viewer import S1PViewer

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    s1p_viewer = S1PViewer(FLAGS.file)
    s1p_viewer.plot_s11_magnitude()
    s1p_viewer.plot_z()


if __name__ == "__main__":
    flags.DEFINE_string("file", None, "S1P file.")
    flags.mark_flag_as_required("file")

    app.run(main)
