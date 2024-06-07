import matplotlib.pyplot as plt
import scienceplots
from absl import app, flags, logging

from simulation.differential_mesh.differential_mesh_graph_factory import \
    DifferentialMeshGraphFactory
from simulation.differential_mesh.differential_mesh_solver import (
    DIFFERENTIAL_MESH_SOLVERS, DifferentialMeshSolver)

FLAGS = flags.FLAGS


def _log_node_potentials(potentials: list[int, float]) -> None:
    """Logs the node potentials.

    Args:
        potentials: List of node potentials.
    """
    logging.info("Node potentials:")
    for node, potential in potentials:
        logging.info("%d %f", node, potential)


def _log_edge_measurements(measurements: list[tuple[int, int], float]) -> None:
    """Logs the differential edge measurements.

    Args:
        measurements: List of edge measurements.
    """
    logging.info("Edge measurements:")
    for (u, v), measurement in measurements:
        logging.info("(%d %d) %f", u, v, measurement)


def solve_differential_mesh_grid(solver_cls: DifferentialMeshSolver,
                                 edge_list: str, noise: float,
                                 verbose: bool) -> None:
    """Solves the differential mesh grid.

    Args:
        solver_cls: Differential mesh solver class.
        edge_list: Edge list filename.
        noise: Standard deviation of the noise.
        verbose: If true, log verbose messages.
    """
    grid = DifferentialMeshGraphFactory.create_from_edge_list(edge_list)
    grid.add_edge_measurement_noise(noise)
    solver = solver_cls(grid, verbose=verbose)
    solver.solve()

    # Log the edge measurements and node potentials.
    _log_edge_measurements(grid.get_edge_measurements())
    _log_node_potentials(grid.get_node_potentials())
    logging.info("MSE = %f", solver.calculate_mean_squared_error())

    # Draw the grid.
    plt.style.use("science")
    grid.draw()


def main(argv):
    assert len(argv) == 1

    solve_differential_mesh_grid(DIFFERENTIAL_MESH_SOLVERS[FLAGS.solver],
                                 FLAGS.edgelist, FLAGS.noise, FLAGS.verbose)


if __name__ == "__main__":
    flags.DEFINE_enum("solver", "matrix", DIFFERENTIAL_MESH_SOLVERS.keys(),
                      "Differential mesh solver.")
    flags.DEFINE_string(
        "edgelist", "simulation/differential_mesh/data/example_2x2.edgelist",
        "Input edge list.")
    flags.DEFINE_float("noise", 0, "Standard deviation of the added noise.")
    flags.DEFINE_boolean("verbose", True, "If true, log verbose messages.")

    app.run(main)
