---
title: 'pat-pde-opt: Differentiable, pattern-forming PDEs for machine learning, optimization, and control'
tags:
  - Python
  - JAX
  - pattern formation
  - differentiable physics
  - partial differential equations
authors:
  - name: Alexander E. Cohen
    orcid: 0000-0002-5284-6775
    corresponding: true
    affiliation: 1
  - name: Sam Degnan-Morgenstern
    affiliation: 1
  - name: Simon Daubner
    orcid: 0000-0002-7944-6026
    affiliation: 2
  - name: Jorn Dunkel
    affiliation: 1
  - name: Martin Z. Bazant
    affiliation: 1
affiliations:
 - name: Massachusetts Institute of Technology, United States
   index: 1
 - name: Imperial College London, United Kingdom
   index: 2
date: 01 October 2025
bibliography: paper.bib
---

# Summary
Pattern formation occurs in many physical systems across many length scales, from quantum systems to planetary atmospheres, and has applications in nanotechnology and biology (cite).
The physical laws and dynamics that dictate pattern formation are often expressed in the form of partial differential equations (PDEs).
To learn models for these systems, optimize these systems, control these systems, or apply any of the numerous methods from machine learning to these systems, we need fast, differentiable, and GPU powered numerical solvers.

The `pat-pde-opt` package provides implementations of PDEs that describe pattern formation in a range of physical systems.
The package aso provides a framework for extending the implementation to more PDEs.
The code is written in JAX and is fully differentiable and GPU-accelerated.
To solve the time-dependent PDEs, `pat-pde-opt` the method of lines and evolves the resulting system of ordinary differential equations (ODEs) using `diffrax` (cite). 
Notably, `diffrax` provides recursively checkpointed adjoint methods which is often essential for differentiating through PDE solves.
`pat-pde-opt` also provides implementations of specialized time stepping methods for many pattern forming systems, including semi-implicit Fourier methods, Strang splitting, and the stabilized explicit Runge-Kutta method ROCK2 (cite).
In addition, the package also provides implementations of the smoothed boundary method for solving PDEs in complex geometries (cite).
The goal of this package is to provide specialized code for the integration of physical simulations of pattern formation with inverse design, optimization, machine learning, and control. 

The `pat-pde-opt` package is organized around Domains, Equations, and Solvers.
The Domain class sets up the computational region for the simulation, including the mesh and axes in both real and Fourier space.
The Domain also stores the Shape, which is used in the context of the smoothed boundary method.
The Shape is initialized with a binary mask where 1s indicate the geometry and 0s indicate the empty space. 
Upon initialization, a smoothed shape required for the smoothed boundary method is produced by evolving an Allen-Cahn equation with a laplacian term with reduced curvature minimization to maintain sharply curved features of the original shape.
The Equation class class consists of implementations of the right hand side of the various PDEs. 
There is a module for each class of PDE and within each module there exists different implementations for variations such as the dimensionality, parameters in the model, and other factors.
The solvers are subclasses of `diffrax.AbstractSolver` and provide implementations of specialized time stepping methods for specific PDEs.
The specification of a Domain, Equation, and a Solver is required for all downstream use cases of `pat-pde-opt`.

Two interfaces are provided through the `PDEModel` and `PDEEnv` class for integrating with machine learning methods.
The `PDEModel` is initialized with a combination of a Domain, Equation, and Solver, and provides three main methods.
The first is a `solve` method for solving the specified Equation on the given Domain with the specified Solver.
In addition, the `fit` method provides the utilities for fitting parameters of functions within the Equation to a dataset.
The `fit` method uses a multiple shooting approach which we found to be both computationally faster and more robust to noise in the dataset than other approaches.
The multiple shooting approach works by `vmap`ing over multiple starting points in the dataset and evaluating the loss at future time points relative to each starting point. 
The specific starting points and residual evaluation points are specified through the `inds` argument of the `fit` function.
Currently, the optimization can be performed using the Levenberg-Marquardt method of BFGS, whose implementations are provided through the `optimistix` package.
Gradients can be computed using forward or reverse mode automatic differentiation, which scale differently with the number of parameters (cite figure and cite julia paper).
The Levenberg-Marquardt method requires Jacobians of the residuals for the Hessian approximation, which can be computed using forward mode automatic differentiation.
Since the Levenberg-Marquardt method uses Gauss-Newton approximations of the Hessian, the method generally converges faster than BFGS. 
However, BFGS does not require the Jacobian of the residuals for the Hessian approximation and thus the gradients can be computed easily using reverse-mode automatic differentiation, which scales much better with the number of parameters.
This tradeoff between scaling with parameter numbers and convergence of optimization must be considered for these differentiable physics optimization problems.
Finally, the `optimize` method uses BFGS from the `optimistix` package to minimize a scalar function of the solution, which is specified in the `objective_function` argument.
`PDEModel` makes it easy to perform model learning and optimization simply from a Domain, Equation, and Solver.

The `PDEEnv` class is useful for turning a PDE into a `Gymnasium`-registered reinforcement learning (RL) environment that can be used to train RL agents with libraries like Stable Baselines (cite pytorch and jax versions).
In addition to the Domain, Equation, and Solver, the `PDEEnv` class requires a `step_dt`, which is the time span of one step of the environment, and a `numeric_dt` which is the time step to use for numerical integration. 
These arae separate parameters because the action time of the agent is often larger than the time step needed for numerical stability.
Beyond these fields, many other pieces of informataion must be provided to form the RL environment, including reward functions, observation functions, and reset functions.

# Statement of need
Mention visualPDE, py-pde, evoxels, dedalus, fenics, dolfinx, google research stuff on the weather code

# Acknowledgements
The authors acknowledge the MIT Office of Research Computing and Data for providing computational resources and advice on open-source scientific computing software.

# References