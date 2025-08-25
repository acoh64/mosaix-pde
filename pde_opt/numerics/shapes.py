import dataclasses

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import dataclasses
import diffrax as dfx

from .utils.derivatives import _gradx_c, _grady_c, _grad2x_c, _grad2y_c, _grad2xy_c

Array = jax.Array


@dataclasses.dataclass
class Shape:
    """Sets up a geometry/shape for solving PDE on with smoothed boundary method.

    The user creates a shape by providing a binary representation and an optional smoothing parameter.
    """

    binary: Array
    dx: Optional[Tuple[float, float]] = (1.0, 1.0)
    smooth_epsilon: float = 1.0
    smooth_curvature: float = 0.0
    smooth_dt: float = 0.1
    smooth_tf: float = 1.0

    def __post_init__(self):
        self.smooth = self.smooth_shape()
        self.smooth = jnp.where(self.smooth < 0.001, 0.001, self.smooth)
        self.smooth = jnp.where(self.smooth > 0.99, 1.0, self.smooth)

    def smooth_shape(self) -> Array:
        """Smooths the shape using the Allen-Cahn equation with curvature minimization."""

        potential = lambda u: 18.0 / self.smooth_epsilon * u * (1.0 - u) * (1.0 - 2.0 * u)

        @jax.jit
        def rhs(t, u, args):
            gradx = _gradx_c(u, self.dx[0])
            grady = _grady_c(u, self.dx[1])
            grad2x = _grad2x_c(u, self.dx[0])
            grad2y = _grad2y_c(u, self.dx[1])
            grad2xy = _grad2xy_c(u, self.dx[0], self.dx[1])
            grad_norm_sq = gradx**2 + grady**2
            grad_norm_sq = jnp.where(grad_norm_sq < 1e-7, 1.0, grad_norm_sq)
            norm_laplace = (
                grad2x * gradx**2 + 2.0 * grad2xy * gradx * grady + grad2y * grady**2
            ) / grad_norm_sq
            laplace = grad2x + grad2y
            return (
                2.0
                * (
                    self.smooth_curvature * laplace
                    + (1.0 - self.smooth_curvature) * norm_laplace
                )
                - potential(u) / self.smooth_epsilon
            )

        solution = dfx.diffeqsolve(
            dfx.ODETerm(rhs),
            dfx.Tsit5(),
            t0=0.0,
            t1=self.smooth_tf,
            dt0=self.smooth_dt,
            y0=self.binary,
            stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-6),
            saveat=dfx.SaveAt(t1=True),
            max_steps=1000000,
        )

        return solution.ys[-1]

    def get_shape_modes(self, N: Optional[int] = None):
        """Get the first N eigenvectors of the graph Laplacian of the binary mask.

        Creates a graph where nodes are the 1-valued pixels, with edges between
        adjacent pixels (left, right, top, bottom neighbors).

        Args:
            N: Number of eigenvectors to return. If None, returns all eigenvectors.

        Returns:
            Array of shape (num_nodes, N) containing the first N eigenvectors
        """
        # Get indices of 1-valued pixels
        nodes = jnp.argwhere(self.binary > 0.5)
        num_nodes = len(nodes)

        if N is None:
            N = num_nodes

        # Build adjacency matrix
        adj = jnp.zeros((num_nodes, num_nodes))

        # Check each node's neighbors
        for i in range(num_nodes):
            node = nodes[i]
            # Check left, right, top, bottom neighbors
            neighbors = [
                [node[0] - 1, node[1]],
                [node[0] + 1, node[1]],
                [node[0], node[1] - 1],
                [node[0], node[1] + 1],
            ]

            for n in neighbors:
                # Find if neighbor exists in nodes list
                n = jnp.array(n)
                mask = (nodes == n).all(axis=1)
                j = jnp.where(mask)[0]
                if len(j) > 0:
                    adj = adj.at[i, j[0]].set(1)
                    adj = adj.at[j[0], i].set(1)

        # Compute graph Laplacian
        degree = jnp.sum(adj, axis=1)
        degree_mat = jnp.diag(degree)
        laplacian = degree_mat - adj

        # Get eigenvectors
        eigenvals, eigenvecs = jnp.linalg.eigh(laplacian)

        # Initialize output array with zeros
        shape = self.binary.shape
        output = jnp.zeros((shape[0], shape[1], N))

        # Fill in eigenvector values at node locations
        for i in range(N):
            eigenvec = eigenvecs[:, i]
            for node_idx, node_pos in enumerate(nodes):
                output = output.at[node_pos[0], node_pos[1], i].set(eigenvec[node_idx])

        self.shape_basis = output
