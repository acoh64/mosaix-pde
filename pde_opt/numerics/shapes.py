"""
This module contains the Shape class, which is used to set up a geometry/shape for solving PDE on with smoothed boundary method.
"""

import dataclasses

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import dataclasses
import diffrax as dfx
import scipy
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

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



    def laplacian_from_mask(self, periodic: bool = False):
        """
        Unnormalized graph Laplacian (4-neighbour) from a 0/1 mask.
        Nodes are entries where mask==1. Two nodes connect if they are
        up/down/left/right neighbours and both are 1.

        Returns:
            L  : (n_nodes, n_nodes) CSR Laplacian
            ids: (H, W) array, node index in [0, n_nodes) or -1 if not a node
        """
        mask = (self.binary > 0)
        H, W = mask.shape
        ids = -np.ones((H, W), dtype=np.int64)
        ids[mask] = np.arange(mask.sum(), dtype=np.int64)
        n = int(mask.sum())
        if n == 0:
            return csr_matrix((0, 0)), ids

        def undirected_edges(dy, dx):
            """Return endpoints (u,v) for each undirected edge, listed once."""
            if periodic:
                m_both = mask & np.roll(mask, (dy, dx), axis=(0, 1))
                if not m_both.any():
                    return np.empty(0, np.int64), np.empty(0, np.int64)
                u = ids[m_both]
                v = np.roll(ids, (dy, dx), axis=(0, 1))[m_both]
                return u, v
            else:
                y0, y1 = max(0, dy), H + min(0, dy)
                x0, x1 = max(0, dx), W + min(0, dx)
                m1 = mask[y0:y1, x0:x1]
                m2 = mask[y0-dy:y1-dy, x0-dx:x1-dx]
                both = m1 & m2
                if not both.any():
                    return np.empty(0, np.int64), np.empty(0, np.int64)
                u = ids[y0:y1, x0:x1][both]
                v = ids[y0-dy:y1-dy, x0-dx:x1-dx][both]
                return u, v

        # Build edges once using right and down neighbours, then symmetrize
        ur, vr = undirected_edges(0, +1)   # right
        ud, vd = undirected_edges(+1, 0)   # down

        u_one = np.concatenate([ur, ud])
        v_one = np.concatenate([vr, vd])

        # Degree from unique undirected edges: each endpoint counted once
        deg = np.bincount(np.concatenate([u_one, v_one]), minlength=n).astype(np.float64)

        # Off-diagonals: symmetrize edges (u,v) and (v,u)
        rows_off = np.concatenate([u_one, v_one])
        cols_off = np.concatenate([v_one, u_one])
        data_off = -np.ones(rows_off.shape[0], dtype=np.float64)

        # Diagonal
        rows = np.concatenate([rows_off, np.arange(n)])
        cols = np.concatenate([cols_off, np.arange(n)])
        data = np.concatenate([data_off, deg])

        L = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        return L, ids


    def get_shape_modes(self, N: Optional[int] = None):
        """Get the first N eigenvectors of the graph Laplacian of the binary mask.

        Creates a graph where nodes are the 1-valued pixels, with edges between
        adjacent pixels (left, right, top, bottom neighbors).

        Args:
            N: Number of eigenvectors to return. If None, returns all eigenvectors.
            downsampling_factor: If provided, downsample binary by this factor before
                computing modes, then upsample results back to original size.
                This can significantly reduce memory usage and computation time
                for large binary masks.

        Returns:
            Array of shape (num_nodes, N) containing the first N eigenvectors
        """
        
        
        laplacian, node_ids = self.laplacian_from_mask()
        # return laplacian, node_ids

        n = laplacian.shape[0]

        # Check if Laplacian matrix is symmetric
        is_symmetric = (laplacian != laplacian.T).nnz == 0
        if not is_symmetric:
            raise ValueError("Laplacian matrix is not symmetric")

        # A scale-aware tiny shift: ~ 1e-8 times a typical diagonal magnitude
        diag_mean = float(laplacian.diagonal().mean()) if n > 0 else 1.0
        sigma = max(diag_mean, 1.0) * 1e-8
        # Get only the first N eigenvectors (much faster than computing all)
        eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(
            laplacian, 
            k=N,
            which='LM',
            sigma=sigma,
            tol=1e-8,
            maxiter=None,
        )

        # Initialize output array with zeros
        shape = self.binary.shape
        output = np.zeros((shape[0], shape[1], N))

        # Vectorized assignment using advanced indexing
        # Get valid node positions (where node_ids >= 0)
        valid_mask = node_ids >= 0
        valid_node_ids = node_ids[valid_mask]
        
        # print(valid_mask)
        # print(valid_node_ids)

        # Fill in eigenvector values at node locations
        for i in range(N):
            eigenvec = eigenvecs[:, i]
            output[valid_mask, i] = eigenvec[valid_node_ids]

        self.shape_basis = jnp.array(output)
        self.shape_basis_evals = eigenvals