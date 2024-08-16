import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from absl import app, flags, logging

from simulation.differential_mesh.differential_mesh_graph_factory import \
    DifferentialMeshGraphFactory
from simulation.differential_mesh.differential_mesh_simulator import \
    DifferentialMeshSimulator
from simulation.differential_mesh.differential_mesh_solver import (
    DIFFERENTIAL_MESH_SOLVERS, DifferentialMeshSolver)
from utils.visualization.color_maps import COLOR_MAPS

FLAGS = flags.FLAGS


def _2d_list_to_grid(values: list[tuple[int, int]],
                     node_to_index_map: dict[int, int], num_rows: int,
                     num_cols: int) -> np.ndarray:
    """Places the values in the list of 2-tuples into a 2D grid.

    The list should consist of 2-tuples, each consisting of the node label and
    the corresponding value.

    Args:
        values: List of 2-tuples.
        node_to_index_map: Map from the node to its index.
        num_rows: Number of rows.
        num_cols: Number of columns.

    Returns:
        A 2D array with the values of the list of 2-tuples.
    """
    grid = np.zeros((num_rows, num_cols))
    for node, value in values:
        node_index = node_to_index_map[node]
        row = node_index // num_cols
        col = node_index % num_cols
        grid[row, col] = value
    return grid


def simulate_standard_error(solver: DifferentialMeshSolver, num_rows: int,
                            num_cols: int, noise: float, num_iterations: int,
                            verbose: bool) -> None:
    """Simulates the standard error of the node potentials.

    Args:
        solver: Differential mesh solver class.
        num_rows: Number of rows.
        num_cols: Number of columns.
        noise: Standard deviation of the noise.
        num_iterations: Number of iterations to simulate.
        verbose: If true, log verbose messages.
    """
    grid = DifferentialMeshGraphFactory.create_zero_2d_graph(num_rows, num_cols)
    simulator = DifferentialMeshSimulator(grid)
    simulated_stderrs = simulator.simulate_node_standard_errors(
        solver, noise, num_iterations, verbose)
    logging.info("Node potential standard errors:")
    for node, stderr in simulated_stderrs:
        logging.info("%d %f", node, stderr)
    calculated_stderrs = grid.calculate_node_standard_errors(noise)

    node_to_index_map = grid.get_node_to_index_map()
    simulated_stderrs_grid = _2d_list_to_grid(simulated_stderrs,
                                              node_to_index_map, num_rows,
                                              num_cols)
    calculated_stderrs_grid = _2d_list_to_grid(calculated_stderrs,
                                               node_to_index_map, num_rows,
                                               num_cols)

    # Plot the simulated and calculated standard error across the grid.
    plt.style.use("science")
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": "3d"},
    )
    surf = ax.plot_surface(
        *np.meshgrid(np.arange(1, num_rows + 1),
                     np.arange(1, num_cols + 1),
                     indexing="ij"),
        simulated_stderrs_grid,
        cmap=COLOR_MAPS["parula"],
        antialiased=False,
    )
    ax.plot_surface(
        *np.meshgrid(np.arange(1, num_rows + 1),
                     np.arange(1, num_cols + 1),
                     indexing="ij"),
        calculated_stderrs_grid,
        cmap=COLOR_MAPS["parula"],
        alpha=0.2,
        antialiased=False,
    )
    ax.set_xlabel("Row")
    ax.set_ylabel("Column")
    ax.set_zlabel("Standard error")
    ax.view_init(30, -45)
    plt.colorbar(surf)
    plt.show()


def simulate_standard_error_sweep(solver: DifferentialMeshSolver,
                                  max_num_rows: int, max_num_cols: int,
                                  noise: float, num_iterations: int,
                                  verbose: bool) -> None:
    """Simulates the standard error of the node potentials while sweeping the
    grid dimensions.

    Args:
        solver: Differential mesh solver class.
        max_num_rows: Maximum number of rows.
        max_num_cols: Maximum number of columns.
        stddev: Standard deviation of the noise.
        num_iterations: Number of iterations to simulate.
        verbose: If true, log verbose messages.
    """
    corner_stderrs = np.zeros((max_num_rows, max_num_cols))
    logging.info("Node potential standard errors:")
    for num_rows in range(1, max_num_rows + 1):
        for num_cols in range(1, max_num_cols + 1):
            if num_rows != 1 or num_cols != 1:
                grid = DifferentialMeshGraphFactory.create_zero_2d_graph(
                    num_rows, num_cols)
                simulator = DifferentialMeshSimulator(grid)
                stderrs = simulator.simulate_node_standard_errors(
                    solver, noise, num_iterations, verbose)
                # Record the standard error at the farthest corner of the grid.
                _, corner_stderr = max(stderrs, key=lambda x: x[0])
                corner_stderrs[num_rows - 1, num_cols - 1] = corner_stderr
                logging.info("(%d, %d) %f", num_rows, num_cols, corner_stderr)

    # Plot the standard error as a function of the grid dimensions.
    plt.style.use("science")
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": "3d"},
    )
    surf = ax.plot_surface(
        *np.meshgrid(np.arange(1, max_num_rows + 1),
                     np.arange(1, max_num_cols + 1),
                     indexing="ij"),
        corner_stderrs,
        cmap=COLOR_MAPS["parula"],
        antialiased=False,
    )
    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Number of columns")
    ax.set_zlabel("Standard error")
    ax.view_init(30, -45)
    plt.colorbar(surf)
    plt.show()


def main(argv):
    assert len(argv) == 1

    simulate_standard_error(DIFFERENTIAL_MESH_SOLVERS[FLAGS.solver],
                            FLAGS.num_rows, FLAGS.num_cols, FLAGS.noise,
                            FLAGS.num_iterations, FLAGS.verbose)
    # simulate_standard_error_sweep(DIFFERENTIAL_MESH_SOLVERS[FLAGS.solver],
    #                               FLAGS.max_num_rows, FLAGS.max_num_cols,
    #                               FLAGS.noise, FLAGS.num_iterations,
    #                               FLAGS.verbose)


if __name__ == "__main__":
    flags.DEFINE_enum("solver", "matrix", DIFFERENTIAL_MESH_SOLVERS.keys(),
                      "Differential mesh solver.")
    flags.DEFINE_integer("num_rows", 3, "Number of rows.")
    flags.DEFINE_integer("num_cols", 4, "Number of columns.")
    flags.DEFINE_integer("max_num_rows", 8, "Maximum number of rows.")
    flags.DEFINE_integer("max_num_cols", 8, "Maximum number of columns.")
    flags.DEFINE_float("noise", 1, "Standard deviation of the added noise.")
    flags.DEFINE_integer("num_iterations", 10000,
                         "Number of iterations to simulate.")
    flags.DEFINE_boolean("verbose", False, "If true, log verbose messages.")

    app.run(main)
