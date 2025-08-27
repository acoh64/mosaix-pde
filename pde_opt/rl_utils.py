import jax.numpy as jnp

_TWO_PI = 2.0 * jnp.pi

def density(psi):
    return jnp.abs(psi)**2

def _wrap_to_pi(x):
    # map to (-pi, pi]
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

def detect_vortices(psi, amp_thresh=0.0, tol=0.5):
    """
    Detect quantum vortices by computing phase circulation on each grid cell.

    Args:
        psi: complex array of shape (N, N), wavefunction on a periodic grid.
        amp_thresh: optional density threshold at the cell center (average of the
            four corners) below which detections are suppressed (default 0.0).
            Useful to ignore spurious wraps in nearly-zero-amplitude regions.
        tol: keep only windings with |circulation| >= tol * 2π (default 0.5).
            This protects against numerical noise; set lower if field is very clean.

    Returns:
        dict with:
            - "winding": int array (N, N), integer circulation per cell (…, -2,-1,0,1,2,…)
            - "positions": float array (K, 2) of cell-center coordinates (i+0.5, j+0.5)
              for cells with nonzero winding (periodic indexing).
            - "charges": int array (K,), winding numbers at those positions.
            - "num_vortices": K (number of nonzero cells)
            - "total_topological_charge": sum of windings over all cells
            - "abs_charge_count": sum of |winding| (counts a double-quantized
              vortex as 2, etc.)
    """
    # Phase
    theta = jnp.angle(psi)

    # Forward wrapped differences (periodic)
    dth_x = _wrap_to_pi(jnp.roll(theta, -1, axis=1) - theta)  # along +x edge
    dth_y = _wrap_to_pi(jnp.roll(theta, -1, axis=0) - theta)  # along +y edge

    # Circulation around each plaquette (i,j) in CCW order:
    # right edge at (i,j), top edge at (i,j+1), left edge at (i+1,j), bottom at (i,j)
    circulation = (
        dth_x
        + jnp.roll(dth_y, -1, axis=1)
        - jnp.roll(dth_x, -1, axis=0)
        - dth_y
    )

    # Convert to integer winding; suppress tiny noisy circulations with tol
    n_float = circulation / _TWO_PI
    n_int = jnp.rint(n_float).astype(jnp.int32)
    n_int = jnp.where(jnp.abs(n_float) >= tol, n_int, 0)

    # Optional density mask at cell centers (average of the four corners)
    if amp_thresh > 0.0:
        rho = jnp.abs(psi) ** 2
        rho_cell = 0.25 * (
            rho
            + jnp.roll(rho, -1, axis=0)
            + jnp.roll(rho, -1, axis=1)
            + jnp.roll(rho, (-1, -1), axis=(0, 1))
        )
        n_int = jnp.where(rho_cell >= amp_thresh, n_int, 0)

    # Extract positions (cell centers) and charges
    idx = jnp.argwhere(n_int != 0)
    charges = n_int[n_int != 0]
    # Cell-center coordinates: (i+0.5, j+0.5). You can map these to physical x,y if needed.
    positions = idx.astype(jnp.float32) + 0.5

    return {
        "winding": n_int,
        "positions": positions,
        "charges": charges,
        "num_vortices": idx.shape[0],
        "total_topological_charge": int(charges.sum()),
        "abs_charge_count": int(jnp.abs(charges).sum()),
    }