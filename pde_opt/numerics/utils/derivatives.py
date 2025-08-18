import jax
import jax.numpy as jnp
from typing import Union, Tuple, Optional, Callable, Dict, Any
from .boundary_conditions import BoundaryType
from ..domains import Domain


# Keep the original laplace method for backward compatibility
def laplacian(domain: Domain):

    @jax.jit
    def laplacian_1d(field: jax.Array) -> jax.Array:
        return (jnp.roll(field, -1) + jnp.roll(field, 1) - 2 * field) / domain.dx[
            0
        ] ** 2

    @jax.jit
    def laplacian_2d(field: jax.Array) -> jax.Array:
        return (
            jnp.roll(field, -1, axis=0)
            + jnp.roll(field, 1, axis=0)
            + jnp.roll(field, -1, axis=1)
            + jnp.roll(field, 1, axis=1)
            - 4 * field
        ) / (domain.dx[0] * domain.dx[1])

    @jax.jit
    def laplacian_3d(field: jax.Array) -> jax.Array:
        return (
            jnp.roll(field, -1, axis=0)
            + jnp.roll(field, 1, axis=0)
            + jnp.roll(field, -1, axis=1)
            + jnp.roll(field, 1, axis=1)
            + jnp.roll(field, -1, axis=2)
            + jnp.roll(field, 1, axis=2)
            - 6 * field
        ) / (domain.dx[0] * domain.dx[1] * domain.dx[2])

    if len(domain.dx) == 1:
        return laplacian_1d
    elif len(domain.dx) == 2:
        return laplacian_2d
    elif len(domain.dx) == 3:
        return laplacian_3d
    else:
        raise ValueError(
            f"Laplacian is not implemented for {len(domain.dx)} dimensions"
        )

def gradient(domain: Domain, axis: int):

    @jax.jit
    def gradient(field: jax.Array) -> jax.Array:
        return (jnp.roll(field, -1, axis=axis) - jnp.roll(field, 1, axis=axis)) / (2 * domain.dx[axis])

    return gradient

# def laplacian(
#     dx: Union[float, Tuple[float, ...]] = 1.0,
#     boundary_conditions: Optional[Dict[str, Any]] = None,
# ) -> Callable[[jax.Array, Optional[float]], jax.Array]:
#     """
#     Create a Laplacian operator function.

#     Args:
#         dx: Grid spacing. Can be a scalar (uniform spacing) or tuple of scalars
#             for each dimension (non-uniform spacing)
#         boundary_conditions: Dictionary specifying boundary conditions for each dimension.
#             Format: {axis: (boundary_type, boundary_values)} where:
#             - axis: int (0, 1, 2 for x, y, z dimensions)
#             - boundary_type: str ('periodic', 'dirichlet', 'neumann', 'time_dependent_dirichlet')
#             - boundary_values: depends on boundary_type
#                 - 'dirichlet':
#                     - float: Constant value for both boundaries
#                     - tuple: (left_val, right_val) for different boundary values
#                     - callable: Function f(pos) that returns boundary value at position
#                     - jax.Array: Spatially dependent boundary values
#                 - 'time_dependent_dirichlet': callable f(time) that returns boundary values
#                 - 'neumann': float or (left_grad, right_grad)
#                 - 'periodic': None (ignored)

#     Returns:
#         Callable function that computes the Laplacian of a JAX array.
#         The function signature is: laplacian_op(field, time=None)

#     Example:
#         >>> import jax.numpy as jnp
#         >>> # Create Laplacian operator with periodic BCs
#         >>> lap_op = laplacian(dx=0.1, boundary_conditions={0: ('periodic', None)})
#         >>> field = jnp.sin(2 * jnp.pi * jnp.linspace(0, 1, 10))
#         >>> result = lap_op(field)

#         # Spatially dependent Dirichlet BCs
#         >>> def boundary_func(pos):
#         ...     return jnp.sin(2 * jnp.pi * pos)
#         >>> lap_op = laplacian(dx=0.1, boundary_conditions={0: ('dirichlet', boundary_func)})

#         # Vector-based boundary values
#         >>> boundary_values = jnp.array([1.0, 2.0, 3.0])  # for 1D field
#         >>> lap_op = laplacian(dx=0.1, boundary_conditions={0: ('dirichlet', boundary_values)})

#         # Time-dependent Dirichlet BCs
#         >>> def time_boundary_func(t):
#         ...     return jnp.array([jnp.sin(t), jnp.cos(t), jnp.sin(2*t)])
#         >>> lap_op = laplacian(dx=0.1, boundary_conditions={0: ('time_dependent_dirichlet', time_boundary_func)})
#         >>> result = lap_op(field, time=1.5)
#     """
#     # Default boundary conditions (zero-padding)
#     if boundary_conditions is None:
#         boundary_conditions = {}

#     def _laplacian_operator(
#         field: jax.Array, time: Optional[float] = None
#     ) -> jax.Array:
#         """Compute the Laplacian of the input field."""
#         if field.ndim not in [1, 2, 3]:
#             raise ValueError(f"Field must be 1D, 2D, or 3D, got {field.ndim}D")

#         # Handle dx parameter
#         if isinstance(dx, (int, float)):
#             dx_array = (dx,) * field.ndim
#         elif len(dx) != field.ndim:
#             raise ValueError(f"dx must have length {field.ndim}, got {len(dx)}")
#         else:
#             dx_array = dx

#         # Convert to JAX arrays
#         dx_array = jnp.array(dx_array)
#         dx2 = dx_array**2

#         if field.ndim == 1:
#             return _compute_1d_laplacian(field, dx2[0], boundary_conditions, time)
#         elif field.ndim == 2:
#             return _compute_2d_laplacian(field, dx2, boundary_conditions, time)
#         else:  # 3D
#             return _compute_3d_laplacian(field, dx2, boundary_conditions, time)

#     return _laplacian_operator


# def _compute_1d_laplacian(
#     field: jax.Array,
#     dx2: float,
#     boundary_conditions: Dict[str, Any],
#     time: Optional[float] = None,
# ) -> jax.Array:
#     """Compute 1D Laplacian using central finite differences."""
#     axis = 0
#     boundary_config = boundary_conditions.get(axis, ("dirichlet", 0.0))
#     boundary_type, boundary_values = boundary_config

#     # Get padded field with boundary conditions
#     from .boundary_conditions import apply_boundary_conditions

#     padded_field = apply_boundary_conditions(
#         field, boundary_type, boundary_values, axis, time
#     )

#     # Use efficient slicing for finite differences
#     left = padded_field[:-2]  # slice(0, -2)
#     right = padded_field[2:]  # slice(2, None)
#     center = padded_field[1:-1]  # slice(1, -1) - this is the original field

#     return (left + right - 2 * center) / dx2


# def _compute_2d_laplacian(
#     field: jax.Array,
#     dx2: Tuple[float, float],
#     boundary_conditions: Dict[str, Any],
#     time: Optional[float] = None,
# ) -> jax.Array:
#     """Compute 2D Laplacian using central finite differences."""
#     dx2_x, dx2_y = dx2

#     # Get boundary configurations for each axis
#     x_bc = boundary_conditions.get(1, ("dirichlet", 0.0))  # x-axis (axis=1)
#     y_bc = boundary_conditions.get(0, ("dirichlet", 0.0))  # y-axis (axis=0)

#     from .boundary_conditions import apply_boundary_conditions

#     # Create a single padded array with boundary conditions for both dimensions
#     # First pad in y-direction (axis=0)
#     padded = apply_boundary_conditions(field, y_bc[0], y_bc[1], axis=0, time=time)
#     # Then pad in x-direction (axis=1)
#     padded = apply_boundary_conditions(padded, x_bc[0], x_bc[1], axis=1, time=time)

#     # Use efficient slicing for finite differences
#     # For x-direction (axis=1)
#     left_x = padded[:, :-2]  # slice(None, slice(0, -2))
#     right_x = padded[:, 2:]  # slice(None, slice(2, None))
#     center_x = padded[:, 1:-1]  # slice(None, slice(1, -1))

#     # For y-direction (axis=0)
#     bottom_y = padded[:-2, :]  # slice(0, -2), slice(None)
#     top_y = padded[2:, :]  # slice(2, None), slice(None)
#     center_y = padded[1:-1, :]  # slice(1, -1), slice(None)

#     return (
#         (left_x + right_x) / dx2_x
#         + (bottom_y + top_y) / dx2_y
#         - 2 * center_x * (1 / dx2_x + 1 / dx2_y)
#     )


# def _compute_3d_laplacian(
#     field: jax.Array,
#     dx2: Tuple[float, float, float],
#     boundary_conditions: Dict[str, Any],
#     time: Optional[float] = None,
# ) -> jax.Array:
#     """Compute 3D Laplacian using central finite differences."""
#     dx2_x, dx2_y, dx2_z = dx2

#     # Get boundary configurations for each axis
#     x_bc = boundary_conditions.get(2, ("dirichlet", 0.0))  # x-axis (axis=2)
#     y_bc = boundary_conditions.get(1, ("dirichlet", 0.0))  # y-axis (axis=1)
#     z_bc = boundary_conditions.get(0, ("dirichlet", 0.0))  # z-axis (axis=0)

#     from .boundary_conditions import apply_boundary_conditions

#     # Create a single padded array with boundary conditions for all dimensions
#     # Apply boundary conditions sequentially for each dimension
#     padded = apply_boundary_conditions(field, z_bc[0], z_bc[1], axis=0, time=time)
#     padded = apply_boundary_conditions(padded, y_bc[0], y_bc[1], axis=1, time=time)
#     padded = apply_boundary_conditions(padded, x_bc[0], x_bc[1], axis=2, time=time)

#     # Use efficient slicing for finite differences
#     # For x-direction (axis=2)
#     left_x = padded[:, :, :-2]  # slice(None, None, slice(0, -2))
#     right_x = padded[:, :, 2:]  # slice(None, None, slice(2, None))
#     center_x = padded[:, :, 1:-1]  # slice(None, None, slice(1, -1))

#     # For y-direction (axis=1)
#     bottom_y = padded[:, :-2, :]  # slice(None, slice(0, -2), None)
#     top_y = padded[:, 2:, :]  # slice(None, slice(2, None), None)
#     center_y = padded[:, 1:-1, :]  # slice(None, slice(1, -1), None)

#     # For z-direction (axis=0)
#     back_z = padded[:-2, :, :]  # slice(0, -2), slice(None, None)
#     front_z = padded[2:, :, :]  # slice(2, None), slice(None, None)
#     center_z = padded[1:-1, :, :]  # slice(1, -1), slice(None, None)

#     return (
#         (left_x + right_x) / dx2_x
#         + (bottom_y + top_y) / dx2_y
#         + (back_z + front_z) / dx2_z
#         - 2 * center_x * (1 / dx2_x + 1 / dx2_y + 1 / dx2_z)
#     )
