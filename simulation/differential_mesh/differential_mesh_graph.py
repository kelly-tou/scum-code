"""The differential mesh graph is an interface for a directed graph, where the
directed edge weights between incident nodes represent the differential
measurements.

The weight of the edge (i, j) represents the differential measurement V(i) - V(j).
The node potential is stored as a node attribute.
"""

import itertools
from collections.abc import Iterator
from enum import Enum, auto
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Root node.
DIFFERENTIAL_MESH_GRAPH_ROOT_NODE = 1

# Node attribute for the node potential.
DIFFERENTIAL_MESH_GRAPH_NODE_POTENTIAL_ATTRIBUTE = "value"

# Node attribute for the node position.
DIFFERENTIAL_MESH_GRAPH_NODE_POSITION_ATTRIBUTE = "pos"

# Edge attribute for the differential measurement.
DIFFERENTIAL_MESH_GRAPH_EDGE_MEASUREMENT_ATTRIBUTE = "weight"

# Default edge drawing color.
DIFFERENTIAL_MESH_GRAPH_EDGE_COLOR = "#1f78b4"


class DifferentialMeshGraphBoundaryCondition(Enum):
    """Boundary condition enumeration."""
    REFERENCE_NODE = auto()
    ZERO_MEAN_NODE_POTENTIAL = auto()


class DifferentialMeshGraph:
    """Interface for a differential mesh graph.

    Attributes:
        graph: Graph object corresponding to the differential mesh graph.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph.copy()
        self._validate_graph()

    def get_node_potentials(self) -> list[tuple[int, float]]:
        """Returns the solved node potentials.

        This function returns a list of 2-tuples, each consisting of the node
        label and its potential.
        """
        return [
            (node, self.get_node_potential(node)) for node in self.graph.nodes
        ]

    def get_node_potential(self, node: int) -> float:
        """Returns the node potential.

        Args:
            node: Node in the differential mesh graph.
        """
        return self.graph.nodes[node].get(
            DIFFERENTIAL_MESH_GRAPH_NODE_POTENTIAL_ATTRIBUTE, 0)

    def set_node_potential(self, node: int, potential: float) -> None:
        """Sets the node potential.

        Args:
            node: Node in the differential mesh graph.
            potential: Potential to set the node to.
        """
        self.graph.nodes[node][
            DIFFERENTIAL_MESH_GRAPH_NODE_POTENTIAL_ATTRIBUTE] = potential

    def reset_node_potentials(self) -> None:
        """Sets potential of all nodes to zero."""
        for node in self.graph.nodes:
            self.set_node_potential(node, 0)

    def get_node_to_index_map(self) -> dict[int, int]:
        """Returns a map from the node to its index."""
        return {
            node: node_index for node_index, node in enumerate(self.graph.nodes)
        }

    def get_index_to_node_map(self) -> dict[int, int]:
        """Returns a map from the index to the node."""
        return {
            node_index: node for node_index, node in enumerate(self.graph.nodes)
        }

    def get_edge_measurements(self) -> list[tuple[tuple[int, int], float]]:
        """Returns the differential edge measurements.

        This function returns a list of 2-tuples, each consisting of a 2-tuple
        denoting the incident nodes of the edge and the differential edge
        measurement.
        """
        return [((u, v), self.get_edge_measurement(u, v))
                for u, v in self.graph.edges]

    def get_edge_measurement(self, u: int, v: int) -> float:
        """Returns the differential edge measurement along the given edge.

        Args:
            u: Outgoing node.
            v: Incoming node.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self.graph.has_edge(u, v):
            raise KeyError(f"({u}, {v})")
        return self.graph.edges[
            u, v][DIFFERENTIAL_MESH_GRAPH_EDGE_MEASUREMENT_ATTRIBUTE]

    def add_edge_measurement_noise(self, stddev: float) -> None:
        """Adds noise to the edge measurements.

        Args:
            stddev: Standard deviation of the added noise.
        """
        for _, _, data in self.graph.edges(data=True):
            data[
                DIFFERENTIAL_MESH_GRAPH_EDGE_MEASUREMENT_ATTRIBUTE] += np.random.normal(
                    scale=stddev)

    def get_neighbors(self, node: int) -> Iterator[int]:
        """Returns an iterator over the neighbors of the given node.

        Args:
            node: Node for which to return the neighbors.
        """
        return itertools.chain(self.graph.predecessors(node),
                               self.graph.successors(node))

    def get_incident_edges(self, node: int) -> Iterator[int]:
        """Returns an iterator over the incident edges of the given node.

        Args:
            node: Node for which to return the neighbors.
        """
        return itertools.chain(self.graph.in_edges(node, data=True),
                               self.graph.out_edges(node, data=True))

    def draw(self) -> None:
        """Draws the differential mesh graph."""
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.get_node_attributes(
            self.graph, DIFFERENTIAL_MESH_GRAPH_NODE_POSITION_ATTRIBUTE)

        # Round the node potentials for drawing.
        potentials = nx.get_node_attributes(
            self.graph, DIFFERENTIAL_MESH_GRAPH_NODE_POTENTIAL_ATTRIBUTE)
        if potentials:
            self._round_map_values(potentials)
            node_labels = potentials
        else:
            node_labels = {node: node for node in self.graph.nodes}

        # Round the edge measurements for drawing.
        measurements = nx.get_edge_attributes(
            self.graph, DIFFERENTIAL_MESH_GRAPH_EDGE_MEASUREMENT_ATTRIBUTE)
        self._round_map_values(measurements)

        # Draw the graph with node and edge labels.
        nx.draw_networkx_nodes(
            self.graph,
            pos=pos,
            ax=ax,
            node_color="none",
            node_size=1000,
            edgecolors=DIFFERENTIAL_MESH_GRAPH_EDGE_COLOR,
        )
        nx.draw_networkx_labels(
            self.graph,
            pos=pos,
            ax=ax,
            labels=node_labels,
        )
        nx.draw_networkx_edges(
            self.graph,
            pos=pos,
            ax=ax,
            width=2,
            edge_color=DIFFERENTIAL_MESH_GRAPH_EDGE_COLOR,
            arrowsize=20,
            node_size=1000,
        )
        nx.draw_networkx_edge_labels(
            self.graph,
            pos=pos,
            ax=ax,
            edge_labels=measurements,
            font_color=DIFFERENTIAL_MESH_GRAPH_EDGE_COLOR,
        )
        ax.axis("off")
        plt.show()

    def create_laplacian_matrix(self) -> np.ndarray:
        """Creates the modified Laplacian matrix.

        The Laplacian matrix is modified by setting the row corresponding to
        the root node to an elementary basis vector because the root node has
        a potential of zero.

        Returns:
            The modified Laplacian matrix.
        """
        node_to_index_map = self.get_node_to_index_map()
        laplacian_matrix = np.zeros(
            (self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        for u, v in self.graph.edges():
            u_index = node_to_index_map[u]
            v_index = node_to_index_map[v]
            laplacian_matrix[u_index, v_index] -= 1
            laplacian_matrix[v_index, u_index] -= 1
            laplacian_matrix[u_index, u_index] += 1
            laplacian_matrix[v_index, v_index] += 1
        return laplacian_matrix

    def create_edge_measurements_vector(self) -> np.ndarray:
        """Creates a vector consisting of the directed sum of the incident edge
        measurements at each node.

        Returns:
            The vector of sums of incident edge measurements at each node.
        """
        node_to_index_map = self.get_node_to_index_map()
        root_index = node_to_index_map[DIFFERENTIAL_MESH_GRAPH_ROOT_NODE]

        edge_measurements_vector = np.zeros(self.graph.number_of_nodes())
        for node_index, node in enumerate(self.graph.nodes):
            outgoing_edge_measurements = np.sum([
                self.get_edge_measurement(u, v)
                for u, v in self.graph.out_edges(node)
            ])
            incoming_edge_measurements = np.sum([
                self.get_edge_measurement(u, v)
                for u, v in self.graph.in_edges(node)
            ])
            edge_measurements = outgoing_edge_measurements - incoming_edge_measurements
            edge_measurements_vector[node_index] = edge_measurements
        edge_measurements_vector[root_index] = 0
        return edge_measurements_vector

    def output_spice_netlist(self, netlist: str, target_node: int = -1) -> None:
        """Outputs the SPICE netlist corresponding to the differential mesh
        graph.

        Each edge is represented by a 1 Ohm resistor, and the root node is
        grounded. A 1 A current source is connected to the target node, and the
        resulting node potential is equal to the standard error of the node
        potential.

        Args:
            netlist: Netlist file to output.
            node: Target node.
        """
        # By default, the target node is the furthest node from the reference node.
        if target_node not in self.graph.nodes:
            target_node = self.graph.number_of_nodes()

        with open(netlist, "w") as f:
            # SPICE title card.
            f.write(f"* SPICE netlist for differential mesh graph\n")

            # Add a 1 Ohm resistor for each edge.
            resistor_index = 1
            for u, v in self.graph.edges:
                f.write(f"R{resistor_index} {u} {v} {{R}}\n")
                resistor_index += 1

            # Add the reference node.
            f.write(f"V 0 {DIFFERENTIAL_MESH_GRAPH_ROOT_NODE} 0\n")

            # Add a 1 A current source to the target node.
            f.write(f"I 0 {target_node} 1\n")

            # Add a DC analysis, define the resistance, and print the node
            # potential.
            f.write(".dc I 1 1 1\n")
            f.write(".param R=1\n")
            f.write(f".print dc v({target_node})\n")
            f.write(".end\n")

    def calculate_node_standard_errors(
        self,
        noise: float = 1,
        boundary_condition: DifferentialMeshGraphBoundaryCondition = (
            DifferentialMeshGraphBoundaryCondition.REFERENCE_NODE),
    ) -> list[tuple[int, int]]:
        """Calculates the standard error of the node potentials using
        eigendecomposition.

        Args:
            noise: Standard deviation of the noise.

        Returns:
            A list of 2-tuples, each consisting of the node label and the
            corresponding standard error.

        Raises:
            ValueError: If the boundary condition is invalid.
        """
        index_to_node_map = self.get_index_to_node_map()
        L = self.create_laplacian_matrix()
        match boundary_condition:
            case DifferentialMeshGraphBoundaryCondition.REFERENCE_NODE:
                node_to_index_map = self.get_node_to_index_map()
                root_index = node_to_index_map[
                    DIFFERENTIAL_MESH_GRAPH_ROOT_NODE]

                L[root_index, :] = 0
                L[:, root_index] = 0
                Linv = np.linalg.pinv(L, hermitian=True)
                squared_stderrs = np.diag(Linv)
            case DifferentialMeshGraphBoundaryCondition.ZERO_MEAN_NODE_POTENTIAL:
                eigenvalues, eigenvectors = np.linalg.eigh(L)
                zero_index = np.argmin(eigenvalues)

                # Remove the zero eigenvalue and its corresponding eigenvector.
                eigenvectors_squared = eigenvectors**2
                eigenvectors_squared_without_zero = np.delete(
                    eigenvectors_squared, zero_index, axis=1)
                eigenvalues_without_zero = np.delete(eigenvalues, zero_index)

                squared_stderrs = np.sum(eigenvectors_squared_without_zero /
                                         eigenvalues_without_zero,
                                         axis=1).T
            case _:
                raise ValueError("Invalid boundary condition: %s.",
                                 boundary_condition)
        return [(index_to_node_map[node_index], np.sqrt(squared_stderr) * noise)
                for node_index, squared_stderr in enumerate(squared_stderrs)]

    def _validate_graph(self) -> None:
        """Validates the graph.

        The graph should contain integer nodes from 1 to the total number of
        nodes.

        Raises:
            ValueError: If the graph is invalid.
        """
        # Validate the nodes are labeled from 1 to the total number of nodes.
        if min(self.graph.nodes) != DIFFERENTIAL_MESH_GRAPH_ROOT_NODE:
            raise ValueError(f"Minimum node should be labeled "
                             f"{DIFFERENTIAL_MESH_GRAPH_ROOT_NODE}.")
        if max(self.graph.nodes) != self.graph.number_of_nodes():
            raise ValueError(f"Node should be labeled consecutively from "
                             f"{DIFFERENTIAL_MESH_GRAPH_ROOT_NODE}.")

    @staticmethod
    def _round_map_values(input_map: dict[Any, float],
                          decimals: int = 6) -> None:
        """Rounds the map values to the given number of decimals.

        Args:
            a: Input dictionary.
            decimals: Number of decimals to round to.
        """
        for key in input_map:
            input_map[key] = np.round(input_map[key], decimals)
