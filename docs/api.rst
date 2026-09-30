API Reference
=============

The core interfaces combine a domain, equation, and solver for simulation,
parameter fitting, optimization, and reinforcement learning.

Models and environments
-----------------------

.. autoclass:: mosaix_pde.pde_model.PDEModel
   :members:

.. autoclass:: mosaix_pde.pde_env.PDEEnv
   :members:

Domains and shapes
------------------

.. automodule:: mosaix_pde.numerics.domains
   :members:

.. automodule:: mosaix_pde.numerics.shapes
   :members:

Equations
---------

.. automodule:: mosaix_pde.numerics.equations.base_eq
   :members:

.. automodule:: mosaix_pde.numerics.equations.cahn_hilliard
   :members:

.. automodule:: mosaix_pde.numerics.equations.allen_cahn
   :members:

.. automodule:: mosaix_pde.numerics.equations.gross_pitaevskii
   :members:

Solvers
-------

.. automodule:: mosaix_pde.numerics.solvers
    :members:

.. automodule:: mosaix_pde.numerics.solvers_rock2
    :members: ROCK2JAX

Parameter functions
-------------------

.. automodule:: mosaix_pde.numerics.functions.cnn
   :members:

.. automodule:: mosaix_pde.numerics.functions.legendre
   :members:

.. automodule:: mosaix_pde.numerics.functions.mixer_mlp
   :members:
