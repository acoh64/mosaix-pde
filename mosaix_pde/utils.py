"""
This module contains general utility functions.
"""


def check_equation_solver_compatibility(solver_type, equation_type):
    """Check that equation type has all required attributes specified by solver.

    This is a check to ensure that the equation and solver are compatible.

    Raises:
        ValueError: If equation is missing any required attributes from solver.
    """

    # Get required attributes from solver if they exist
    if not hasattr(solver_type, "required_equation_attrs"):
        return

    solver_required_attrs = solver_type.required_equation_attrs

    # Check each required attribute exists in equation class
    missing_attrs = []
    for attr in solver_required_attrs:
        if not hasattr(equation_type, attr):
            missing_attrs.append(attr)

    if missing_attrs:
        raise ValueError(
            f"Equation type {equation_type.__name__} is missing required "
            f"attributes for solver {solver_type.__name__}: {missing_attrs}"
        )


def prepare_solver_params(solver_type, solver_parameters, equation):
    """Prepare solver parameters by extracting required equation attributes.

    Some solvers require attributes from the equation to be passed to them.
    This function prepares the solver parameters by extracting the required attributes from the equation.

    Args:
        solver_parameters (Dict[str, Any]): The solver parameters to use for the equation
        equation (BaseEquation): The equation to solve

    Returns:
        Dict[str, Any]: The prepared solver parameters
    """

    full_solver_params = solver_parameters.copy()
    if hasattr(solver_type, "required_equation_attrs"):
        for attr_name in solver_type.required_equation_attrs:
            full_solver_params[attr_name] = getattr(equation, attr_name)

    return full_solver_params
