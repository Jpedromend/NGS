<img src="logo.png" alt="NGS.jl" width="400">

# NGS: Variational non-Gaussian solutions to interacting spin-boson models

**NGS** is a Julia package designed to study ground-state properties of strongly correlated spin-boson systems. It relies on a hybrid numerical framework that optimizes a non-Gaussian state (NGS) ansatz.

This package implements the method described in the following papers:
>  1. **Variational non-gaussian solutions to interacting spin-boson models**,
> JP Mendonça, Y Wang, and K Jachymski,
> [arXiv link / DOI placeholder] 
> 2. **Role of Matter Interactions in Superradiant Phenomena**,
> JP Mendonça, K Jachymski, Y Wang,
> Physical Review Letters 135 (13), 133601 (2025)

If you use this code in your research, please cite the associated manuscripts (cf. License).

## Overview

The simulation of spin-boson models is often hindered by the infinite-dimensional nature of the bosonic Hilbert space and the presence of strong many-body correlations. Standard approaches typically rely on truncation (limiting photon numbers) or mean-field approximations that neglect entanglement.

This framework addresses these challenges by combining:

1.  **Non-Gaussian Variational Ansatz:**

$$|\psi_{\rm NGS}\rangle = U_{\lambda} \left( U_{\mathrm{GS}} |0\rangle_{\rm b}\otimes|\phi\rangle_{\rm s} \right).$$

The bosonic sector is treated using a variational manifold combining displacement and squeezing, forming a Gaussian state. A (Lang-Firsov-inspired) dressing transformation introduces entanglement between the two subsystems. This replaces explicit photon number truncation.

2.  **Many-Body Solver:** The many-body spin state $|\phi\rangle_{\rm s}$ is obtained, with no further approximations, solving the effective spin Hamiltonian

$$H_{\rm eff} = \langle \psi_{\rm b} | U_\lambda^\dagger H U_\lambda | \psi_{\rm b} \rangle,$$

using Density Matrix Renormalization Group (DMRG) via `ITensors.jl`, capturing spin-spin correlations.

A self-consistent optimization loop then minimizes the variational energy with respect to both the variational parameters and the spin wavefunction, leading to an accurate approximation of the full spin-boson ground state beyond mean-field theory.

## Features

  * **Truncation-free Bosons:** Effectively handles regimes with macroscopic photon occupancy (e.g., superradiance) without convergence issues related to basis size.
  * **Reduced Computational Cost:** The necessary bond-dimension to obtain the ground state is significantly reduced, as the MPS solver only has to deal with an effective spin-only model.
  * **Modular Architecture:** The package is designed to be extensible, utilizing multiple dispatch for models and algorithms. 

## Installation

This package is currently provided as a local research package. To use it:

1.  Clone the repository:

    ```bash
    git clone https://github.com/Jpedromend/NGS.git
    cd NGS
    ```

2.  Instantiate the Julia environment:

    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

## Usage

Below is a minimal example demonstrating the simulation of the Dicke-Ising Model.

```julia
using NGS

# 1. Define the Physics Model
N = 20

# Coupling Example (Nearest-neighbor Z-interaction)
coups = [Couplings(i, i+1, 2.0, :z) for i in 1:N-1]

model = ExtendedDickeModel(
    N=N, 
    omega=1.0, 
    g=0.8,
    epsilon=1.0,
    couplings=coups
)

# 2. Run the Solver
result = solve(model; max_iter=250, tol=1e-8)

# 3. Calculate Observables
avg_n = expect_n(model, result)

sz_sites = expect_sz(model, result)
mz = sum(sz_sites) / N

println("Ground State Energy: ", result.energy)
println("Mean Photon Number:  ", avg_n)
println("Magnetization Mz:    ", mz)

# Accessing Convergence Telemetry
println("Converged in $(result.stats.iterations) iterations.")
```

More examples can be found in `notebooks/`.

## Repository Structure

  * `src/`: Source code for the library.
      * `NGS.jl`: Main module definition and exports.
      * `types.jl`: Core types, model definitions, and the `SolverResult` struct.
      * `physics.jl`: Implementation of the effective Hamiltonian, MPO caching, and energy functionals.
      * `solver.jl`: The self-consistent optimization loops and DMRG interface.
      * `observables.jl`: Independent functions for expected values and correlations.
  * `notebooks/`: Notebook examples for reproducing benchmarks and plotting phase diagrams.

