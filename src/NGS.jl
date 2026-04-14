module NGS

using ITensors, ITensorMPS, Optim, LinearAlgebra, Printf, SparseArrays

# Load scripts
include("types.jl")
include("physics.jl")
include("solver.jl")
include("observables.jl")

# --- Public API ---

# Types
export ExtendedDickeModel, Couplings, SolverResult, SolverStats

# Dispatch Tags
export NGSDMRG, GSDMRG

# Core Methods
export solve, init_env

# Observables
export mean_photon_number, collective_magnetizations, structure_factor_z, spin_boson_correlators

end