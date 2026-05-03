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
export expect_n, expect_sx, expect_sz, correlation_matrix_sxsx, correlation_matrix_szsz

end