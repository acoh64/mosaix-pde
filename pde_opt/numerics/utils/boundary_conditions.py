import jax
import jax.numpy as jnp
from typing import Union, Tuple, Optional, Callable, Any
from enum import Enum

class BoundaryType(Enum):
    """Enumeration of boundary condition types."""
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    TIME_DEPENDENT_DIRICHLET = "time_dependent_dirichlet"

# Shorthands in slicing logic
__ = slice(None)    # all elements [:]
_i_ = slice(1, -1)  # inner elements [1:-1]

CENTER = (__,  _i_, _i_, _i_)
LEFT   = (__, slice(None,-2), _i_, _i_)
RIGHT  = (__, slice(2, None), _i_, _i_)
BOTTOM = (__, _i_, slice(None,-2), _i_)
TOP    = (__, _i_, slice(2, None), _i_)
BACK   = (__, _i_, _i_, slice(None,-2))
FRONT  = (__, _i_, _i_, slice(2, None))

def apply_boundary_conditions(field: jax.Array, 
                            boundary_type: Union[str, BoundaryType],
                            boundary_values: Optional[Union[float, Tuple[float, ...], Callable, jax.Array]] = None,
                            axis: int = 0,
                            time: Optional[float] = None) -> jax.Array:
    """
    Apply boundary conditions to a field along a specified axis.
    
    Args:
        field: Input array
        boundary_type: Type of boundary condition ('periodic', 'dirichlet', 'neumann', 'time_dependent_dirichlet')
        boundary_values: Values for boundary conditions. Can be:
            - float: Constant value for Dirichlet/Neumann
            - tuple: Different values for each boundary (left/right, bottom/top, etc.)
            - callable: Function that takes position and returns boundary value
            - jax.Array: Spatially dependent boundary values (must match boundary shape)
            - callable: For time_dependent_dirichlet, function that takes time and returns boundary values
        axis: Axis along which to apply boundary conditions
        time: Current time (required for time_dependent_dirichlet)
    
    Returns:
        Array with boundary conditions applied
    """
    if isinstance(boundary_type, str):
        boundary_type = BoundaryType(boundary_type)
    
    if boundary_type == BoundaryType.PERIODIC:
        return apply_periodic_boundary(field, axis)
    elif boundary_type == BoundaryType.DIRICHLET:
        return apply_dirichlet_boundary(field, boundary_values, axis)
    elif boundary_type == BoundaryType.NEUMANN:
        return apply_neumann_boundary(field, boundary_values, axis)
    elif boundary_type == BoundaryType.TIME_DEPENDENT_DIRICHLET:
        if time is None:
            raise ValueError("time parameter is required for time_dependent_dirichlet boundary conditions")
        return apply_time_dependent_dirichlet_boundary(field, boundary_values, axis, time)
    else:
        raise ValueError(f"Unknown boundary type: {boundary_type}")

def apply_periodic_boundary(field: jax.Array, axis: int = 0) -> jax.Array:
    """Apply periodic boundary conditions using pad with wrap mode."""
    pad_width = [(0, 0)] * field.ndim
    pad_width[axis] = (1, 1)
    return jnp.pad(field, pad_width, mode='wrap')

def apply_dirichlet_boundary(field: jax.Array, 
                           boundary_values: Union[float, Tuple[float, ...], Callable, jax.Array],
                           axis: int = 0) -> jax.Array:
    """
    Apply Dirichlet boundary conditions using ghost cell approach.
    
    Ghost cell values are computed as: ghost = 2*bc - interior
    where bc is the boundary condition value and interior is the adjacent interior point.
    
    Args:
        field: Input array
        boundary_values: Can be:
            - float: Constant value for both boundaries
            - tuple: (left_val, right_val) for different boundary values
            - callable: Function f(pos) that returns boundary value
                       pos is a scalar (0.0 for left boundary, 1.0 for right boundary)
            - jax.Array: Spatially dependent boundary values
        axis: Axis along which to apply boundary conditions
    """
    # Get boundary values
    if isinstance(boundary_values, (int, float)):
        bc0 = bc1 = boundary_values
    elif isinstance(boundary_values, (tuple, list)):
        if len(boundary_values) == 1:
            bc0 = bc1 = boundary_values[0]
        elif len(boundary_values) == 2:
            bc0, bc1 = boundary_values
        else:
            raise ValueError("boundary_values tuple must have length 1 or 2")
    elif callable(boundary_values):
        bc0 = boundary_values(0.0)  # Left boundary
        bc1 = boundary_values(1.0)  # Right boundary
    elif isinstance(boundary_values, jax.Array):
        # Handle array-based boundary values
        if boundary_values.ndim == 1 and len(boundary_values) == 2:
            bc0, bc1 = boundary_values
        elif boundary_values.ndim == field.ndim - 1:
            bc0 = bc1 = boundary_values
        elif boundary_values.ndim == field.ndim and boundary_values.shape[axis] == 2:
            bc0 = jnp.take(boundary_values, 0, axis=axis)
            bc1 = jnp.take(boundary_values, 1, axis=axis)
        else:
            raise ValueError(f"Boundary array shape {boundary_values.shape} is incompatible with field shape {field.shape}")
    else:
        raise ValueError("boundary_values must be float, tuple, callable, or jax.Array")
    
    # Create padded array
    pad_width = [(0, 0)] * field.ndim
    pad_width[axis] = (1, 1)
    padded = jnp.pad(field, pad_width, mode='constant', constant_values=0)
    
    # Compute ghost cell values using Dirichlet condition: ghost = 2*bc - interior
    # Left ghost cell (index 0)
    left_slices = [slice(None)] * field.ndim
    left_slices[axis] = 0
    interior_left_slices = [slice(None)] * field.ndim
    interior_left_slices[axis] = 1
    padded = padded.at[tuple(left_slices)].set(2.0 * bc0 - padded[tuple(interior_left_slices)])
    
    # Right ghost cell (index -1)
    right_slices = [slice(None)] * field.ndim
    right_slices[axis] = -1
    interior_right_slices = [slice(None)] * field.ndim
    interior_right_slices[axis] = -2
    padded = padded.at[tuple(right_slices)].set(2.0 * bc1 - padded[tuple(interior_right_slices)])
    
    return padded

def apply_time_dependent_dirichlet_boundary(field: jax.Array, 
                                          boundary_func: Callable, 
                                          axis: int = 0, 
                                          time: float = 0.0) -> jax.Array:
    """
    Apply time-dependent Dirichlet boundary conditions.
    
    Args:
        field: Input array
        boundary_func: Function f(time) that returns boundary values at given time
        axis: Axis along which to apply boundary conditions
        time: Current time
    
    Returns:
        Array with time-dependent boundary conditions applied
    """
    # Get boundary values at current time
    boundary_values = boundary_func(time)
    
    # Apply the boundary values using the standard Dirichlet method
    return apply_dirichlet_boundary(field, boundary_values, axis)

def apply_neumann_boundary(field: jax.Array, 
                          boundary_values: Union[float, Tuple[float, ...]],
                          axis: int = 0) -> jax.Array:
    """Apply Neumann boundary conditions (fixed gradients at boundaries)."""
    if isinstance(boundary_values, (int, float)):
        left_grad = right_grad = boundary_values
    elif isinstance(boundary_values, (tuple, list)):
        if len(boundary_values) == 1:
            left_grad = right_grad = boundary_values[0]
        elif len(boundary_values) == 2:
            left_grad, right_grad = boundary_values
        else:
            raise ValueError("boundary_values tuple must have length 1 or 2")
    else:
        raise ValueError("boundary_values must be float or tuple")
    
    # For zero gradient (most common case), use edge mode
    if left_grad == 0 and right_grad == 0:
        pad_width = [(0, 0)] * field.ndim
        pad_width[axis] = (1, 1)
        return jnp.pad(field, pad_width, mode='edge')
    else:
        # For non-zero gradients, we need custom padding
        return _apply_nonzero_neumann_boundary(field, left_grad, right_grad, axis)

def _apply_nonzero_neumann_boundary(field: jax.Array, 
                                  left_grad: float, 
                                  right_grad: float, 
                                  axis: int = 0) -> jax.Array:
    """Apply Neumann boundary conditions with non-zero gradients."""
    # Create padded field
    padded_shape = list(field.shape)
    padded_shape[axis] += 2
    
    padded_field = jnp.zeros(padded_shape, dtype=field.dtype)
    
    # Set left boundary (extrapolate)
    left_slices = [slice(None)] * field.ndim
    left_slices[axis] = 0
    first_slices = [slice(None)] * field.ndim
    first_slices[axis] = 0
    left_boundary = field[tuple(first_slices)] - left_grad
    padded_field = padded_field.at[tuple(left_slices)].set(left_boundary)
    
    # Set right boundary (extrapolate)
    right_slices = [slice(None)] * field.ndim
    right_slices[axis] = -1
    last_slices = [slice(None)] * field.ndim
    last_slices[axis] = -1
    right_boundary = field[tuple(last_slices)] + right_grad
    padded_field = padded_field.at[tuple(right_slices)].set(right_boundary)
    
    # Insert the original field
    slices = [slice(None)] * field.ndim
    slices[axis] = slice(1, -1)
    padded_field = padded_field.at[tuple(slices)].set(field)
    
    return padded_field