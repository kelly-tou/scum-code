import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from absl import app, flags, logging

from simulation.differential_mesh.differential_mesh_graph_factory import \
    DifferentialMeshGraphFactory
from simulation.differential_mesh.differential_mesh_solver import (
    DIFFERENTIAL_MESH_SOLVERS, IterativeDifferentialMeshSolver)
from utils.visualization.color_maps import COLOR_MAPS

FLAGS = flags.FLAGS


def simulate_num_iterations(solver_cls: IterativeDifferentialMeshSolver,
                            max_num_rows: int, max_num_cols: int, noise: float,
                            num_iterations_per_grid: int,
                            output_csv: str) -> None:
    """Simulates the number of iterations to solve the differential mesh grid.

    Args:
        solver_cls: Differential mesh solver class.
        max_num_rows: Maximum number of rows.
        max_num_cols: Maximum number of columns.
        noise: Standard deviation of the noise.
        num_iterations_per_grid: Number of iterations per grid.
        output_csv: Output CSV file.
    """
    with open(output_csv, "w") as f:
        f.write("Number of rows,Number of columns,Number of iterations\n")
        for num_rows in range(1, max_num_rows + 1):
            for num_cols in range(1, max_num_cols + 1):
                if num_rows != 1 or num_cols != 1:
                    for _ in range(num_iterations_per_grid):
                        grid = DifferentialMeshGraphFactory.create_zero_2d_graph(
                            num_rows, num_cols)
                        grid.add_edge_measurement_noise(noise)
                        solver = solver_cls(grid, verbose=False)
                        solver.solve()

                        # Output the number of iterations.
                        num_iterations = solver.num_iterations
                        f.write(f"{num_rows},{num_cols},{num_iterations}\n")
                        logging.info("(%d, %d) %d", num_rows, num_cols,
                                     num_iterations)


def plot_num_iterations(num_iterations_per_grid: str) -> None:
    """Plots the number of iterations to solve the differential mesh grid.

    Args:
        num_iterations_per_grid: CSV file with the number of iterations per
          grid size.
    """
    df = pd.read_csv(num_iterations_per_grid)
    num_rows_column, num_cols_column, num_iterations_column = df.columns
    logging.info(df.describe())

    max_num_rows = df[num_rows_column].max()
    max_num_cols = df[num_cols_column].max()
    mean_num_iterations = df.groupby([num_rows_column, num_cols_column
                                     ])[num_iterations_column].mean()

    logging.info("Mean iterations:")
    num_iterations = np.zeros((max_num_cols, max_num_rows))
    for num_rows in range(1, max_num_rows + 1):
        for num_cols in range(1, max_num_cols + 1):
            if (num_rows, num_cols) in mean_num_iterations.index:
                mean_iterations = mean_num_iterations.loc[(num_rows, num_cols)]
                num_iterations[num_cols - 1, num_rows - 1] = mean_iterations
                logging.info("(%d, %d) %f", num_rows, num_cols, mean_iterations)

    # Plot the number of iterations as a function of the grid dimensions.
    plt.style.use("science")
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": "3d"},
    )
    surf = ax.plot_surface(
        *np.meshgrid(np.arange(1, max_num_rows + 1),
                     np.arange(1, max_num_cols + 1)),
        num_iterations,
        cmap=COLOR_MAPS["parula"],
        antialiased=False,
    )
    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Number of columns")
    ax.set_zlabel("Mean iterations")
    ax.view_init(30, -45)
    plt.colorbar(surf)
    plt.show()


def main(argv):
    assert len(argv) == 1

    # simulate_num_iterations(DIFFERENTIAL_MESH_SOLVERS[FLAGS.solver],
    #                         FLAGS.max_num_rows, FLAGS.max_num_cols, FLAGS.noise,
    #                         FLAGS.num_iterations_per_grid, FLAGS.output_csv)
    plot_num_iterations(FLAGS.output_csv)


if __name__ == "__main__":
    flags.DEFINE_enum("solver", "priority", [
        key for key, cls in DIFFERENTIAL_MESH_SOLVERS.items()
        if issubclass(cls, IterativeDifferentialMeshSolver)
    ], "Iterative differential mesh solver.")
    flags.DEFINE_integer("max_num_rows", 10, "Maximum number of rows.")
    flags.DEFINE_integer("max_num_cols", 10, "Maximum number of columns.")
    flags.DEFINE_float("noise", 0, "Standard deviation of the added noise.")
    flags.DEFINE_integer("num_iterations_per_grid", 100,
                         "Number of iterations per grid.")
    flags.DEFINE_string(
        "output_csv",
        "simulation/differential_mesh/data/num_iterations_20x20.csv",
        "Output CSV file.")

    app.run(main)
