# Examples

This directory contains standalone Python versions of the examples in the project
documentation. Run a script from the repository root after installing `mosaix-pde`:

```bash
python examples/solving_pde.py
```

| Script | Corresponding documentation | Description |
| --- | --- | --- |
| [`readme_example.py`](readme_example.py) | [README usage example](../README.md#usage) | Solve a Cahn--Hilliard equation and train a periodic CNN parameterization. |
| [`solving_pde.py`](solving_pde.py) | [`solving_pde.ipynb`](../docs/notebooks/solving_pde.ipynb) | Solve and animate a periodic two-dimensional Cahn--Hilliard equation. |
| [`solving_pde_smoothed_boundary.py`](solving_pde_smoothed_boundary.py) | [`solving_pde_smoothed_boundary.ipynb`](../docs/notebooks/solving_pde_smoothed_boundary.ipynb) | Solve an Allen--Cahn equation on a smoothed, nonrectangular boundary. |
| [`optimization_neural_network.py`](optimization_neural_network.py) | [`optimization_neural_network.ipynb`](../docs/notebooks/optimization_neural_network.ipynb) | Learn a Cahn--Hilliard chemical potential with a periodic CNN. |
| [`optimization_3d.py`](optimization_3d.py) | [`optimization_3D.ipynb`](../docs/notebooks/optimization_3D.ipynb) | Fit constitutive functions in a three-dimensional Cahn--Hilliard model. |
| [`pde_rl_env.py`](pde_rl_env.py) | [`pde_rl_env.ipynb`](../docs/notebooks/pde_rl_env.ipynb) | Construct and sample a PDE reinforcement-learning environment. |

The optimization, three-dimensional, smoothed-boundary, and reinforcement-learning
examples perform substantial numerical computations and may take several minutes or
longer on CPU. The scripts open Matplotlib windows for their visualizations; the
notebooks provide inline, step-by-step versions of the same workflows.
