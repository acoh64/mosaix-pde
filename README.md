<img src="https://raw.githubusercontent.com/acoh64/mosaix-pde/main/docs/logo.png" width="400em" align="right" />

# mosaix-pde

`mosaix-pde` is a package for optimizing pattern forming PDEs that appear in different areas of physics, written in [JAX](https://github.com/jax-ml/jax). 
It has code for PDE optimization and control with gradient-based methods and reinforcement learning.
We use [diffrax](https://github.com/patrick-kidger/diffrax) for time stepping and implement system-specific solvers, such as semi-implicit Fourier methods and Strang splitting.

You can find the full documentation on [read the docs](https://mosaix-pde.readthedocs.io).

## Installation

`mosaix-pde` requires Python 3.10 or later. Clone the repository and install
the package in a Conda environment:

```bash
git clone https://github.com/acoh64/mosaix-pde.git
cd mosaix-pde
conda create -y -n mosaix-pde-env python=3.12
conda activate mosaix-pde-env
python -m pip install -e .
```

Alternatively, Python's built-in `venv` can be used after cloning the
repository:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

By default, it will install the CPU version of JAX.
To use with GPU, run:
```bash
pip install -U "jax[cuda12]"
```

## Examples and tutorials

Standalone Python examples are available in the [`examples/`](https://github.com/acoh64/mosaix-pde/tree/main/examples)
directory. Each tutorial notebook in [`docs/notebooks/`](https://github.com/acoh64/mosaix-pde/tree/main/docs/notebooks) has a
corresponding script that can be run outside Jupyter. The notebooks are the primary
step-by-step tutorials and are also rendered in the
[online documentation](https://mosaix-pde.readthedocs.io/), while the scripts are
convenient for running and adapting complete examples.

The full workflow shown in the Usage section below is available as
[`examples/readme_example.py`](https://github.com/acoh64/mosaix-pde/blob/main/examples/readme_example.py). See the
[`examples` index](https://github.com/acoh64/mosaix-pde/blob/main/examples/README.md) for a description of every script and its
corresponding notebook.

## Usage

This CPU quickstart uses a 32 × 32 grid, 2,000 integration steps, and a
13-parameter pointwise neural network. We fit it with up to 20 Levenberg–Marquardt
iterations (`method="least_squares"`), which is practical for this small number of
parameters. The example reports trajectory prediction error before and after
training so you can check that the fit improves. The iteration cap keeps the
example short; it does not guarantee optimizer convergence or recovery of the
chemical potential outside the sampled concentrations. The first run also
includes JAX compilation. For a larger CNN trained with BFGS, see the
[neural-network tutorial](https://mosaix-pde.readthedocs.io/en/latest/notebooks/optimization_neural_network.html).

Here is an example of solving the Cahn-Hilliard equation in 2D with periodic boundary conditions using a semi-implicit Fourier method:

```python
import jax
import jax.numpy as jnp

from mosaix_pde import PDEModel
from mosaix_pde import CahnHilliard2DPeriodic
from mosaix_pde import SemiImplicitFourierSpectral
from mosaix_pde import Domain
from mosaix_pde import PeriodicCNN

Nx = Ny = 32
Lx = Ly = 0.01 * Nx

domain = Domain((Nx, Ny), ((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2)), "dimensionless")

opt_model = PDEModel(equation_type=CahnHilliard2DPeriodic, domain=domain, solver_type=SemiImplicitFourierSpectral)

params = {"kappa": 0.002, "mu": lambda c: jnp.log(c / (1.0 - c)) + 3.0 * (1.0 - 2.0 * c), "D": lambda c: c * (1. - c)}

solver_params = {"A": 0.5}

key = jax.random.PRNGKey(0)
y0 = jnp.clip(0.1 * jax.random.normal(key, (Nx, Ny)) + 0.5, 0.01, 0.99)
ts = jnp.linspace(0.0, 0.002, 5)

sol = opt_model.solve(params, y0, ts, solver_params, dt0=0.000001, max_steps=1000000)
```

Next, here is an example of using the previous solution as a dataset to fit a neural network for the chemical potential term:

```python
data = {}
data['ys'] = sol
data['ts'] = ts

model = PeriodicCNN(
    in_channels=1,
    hidden_channels=(4,),
    out_channels=1,
    kernel_size=1,
    key=jax.random.PRNGKey(0),
)

init_params = {"mu": model}
static_params = {"kappa": 0.002, "D": lambda c: c * (1. - c)}
solver_parameters = {"A": 0.5}
weights = {"mu": None}
lambda_reg = 0.0

inds = [[0, 1, 2], [2, 3, 4]]

def trajectory_mse(parameters):
    predicted = opt_model.solve(parameters, y0, ts, solver_parameters)
    return jnp.mean((predicted[1:] - sol[1:]) ** 2)

initial_mse = float(trajectory_mse({**init_params, **static_params}))
res = opt_model.train(data, inds, init_params, static_params, solver_parameters, weights, lambda_reg, method="least_squares", max_steps=20)
final_mse = float(trajectory_mse(res))
print(f"Trajectory MSE before training: {initial_mse:.3e}")
print(f"Trajectory MSE after training:  {final_mse:.3e}")
print(f"Error reduction: {initial_mse / final_mse:.1f}x")
```

## Current Model Implementations

This package is designed to support pattern-forming PDEs across a wide-range of physical systems.
We have currently implemented variants of the following equations:
- Cahn-Hilliard equation
  - 2D with periodic boundary conditions
  - 3D with periodic boundary conditions
  - 2D with smoothed boundary method
- Allen-Cahn equation
  - 2D with periodic boundary conditions
  - 2D with constant current conditions + Butler-Volmer kinetics (for battery applications)
  - 2D with smoothed boundary
  - 2D with smoothed boundar and constant current conditions + Butler-Volmer kinetics (for battery applications)
- Gross-Pitaevskii
  - Reduced 2D with periodic boundary conditions
  - Rotating reduced with 2D periodic boundary conditions

## Running Tests
To run the tests in the `tests/` directory, run 
```bash
pytest tests/
```

## Contributing

Bug reports, feature requests, documentation improvements, and code contributions
are welcome. See [CONTRIBUTING.md](https://github.com/acoh64/mosaix-pde/blob/main/CONTRIBUTING.md) for reporting, support,
development, testing, and pull-request guidelines.

## TODO

- [ ] Arbitrary boundary conditions
- [ ] Implicit time stepping
- [ ] Multi-GPU support
- [ ] Extend to non-Cartesian domains
- [ ] WandB logging and checkpointing

## License

This code has been published under the MIT licence.

## Acknowledgments
This project builds on the excellent JAX ecosystem for scientific computing.
We gratefully acknowledge the following open-source libraries:
- [Diffrax](https://github.com/patrick-kidger/diffrax)
- [Optimistix](https://github.com/patrick-kidger/optimistix)
- [Equinox](https://github.com/patrick-kidger/equinox)

We especially thank [Patrick Kidger](https://github.com/patrick-kidger) for developing amazing JAX software.
