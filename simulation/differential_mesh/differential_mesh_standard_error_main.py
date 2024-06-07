import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from absl import app, flags, logging

from simulation.differential_mesh.differential_mesh_graph_factory import \
    DifferentialMeshGraphFactory
from utils.visualization.color_maps import COLOR_MAPS

FLAGS = flags.FLAGS


def plot_standard_error(num_rows: int, num_cols: int) -> None:
    """Plots the standard error of the node potentials.

    Args:
        num_rows: Number of rows.
        num_cols: Number of columns.
    """
    grid = DifferentialMeshGraphFactory.create_zero_2d_graph(num_rows, num_cols)
    stderrs = grid.calculate_node_standard_errors()
    logging.info("Node potential standard errors:")
    for node, stderr in stderrs:
        logging.info("%d %f", node, stderr)

    node_to_index_map = grid.get_node_to_index_map()
    node_stderrs = np.zeros((num_cols, num_rows))
    for node, stderr in stderrs:
        node_index = node_to_index_map[node]
        row = node_index // num_cols
        col = node_index % num_cols
        node_stderrs[col, row] = stderr

    # Plot the standard error across the grid.
    plt.style.use("science")
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": "3d"},
    )
    surf = ax.plot_surface(
        *np.meshgrid(np.arange(1, num_rows + 1), np.arange(1, num_cols + 1)),
        node_stderrs,
        cmap=COLOR_MAPS["parula"],
        antialiased=False,
    )
    ax.set_xlabel("Row")
    ax.set_ylabel("Column")
    ax.view_init(30, -45)
    plt.colorbar(surf)
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_standard_error(FLAGS.num_rows, FLAGS.num_cols)


if __name__ == "__main__":
    flags.DEFINE_integer("num_rows", 3, "Number of rows.")
    flags.DEFINE_integer("num_cols", 4, "Number of columns.")

    app.run(main)
