# src/types.jl

# --- Core Hierarchy ---
abstract type AbstractSpinBosonModel end
abstract type AbstractAnsatz end
abstract type AbstractSolverBackend end

# --- Numerical Backends ---
struct DMRGBackend <: AbstractSolverBackend end

# --- Explicit Interaction Graphs ---
struct SpinCoupling
    axis::Symbol
    i::Int
    j::Int
    val::Float64
end

struct SpinBosonCoupling
    m::Int
    axis::Symbol 
    site::Int
    val::Float64
end

# --- The Physical Builder ---
mutable struct SpinBosonSystem <: AbstractSpinBosonModel
    N::Int                                 
    omega::Vector{Float64}                 
    epsilon::Vector{Float64}          
    spin_couplings::Vector{SpinCoupling} 
    spin_boson_couplings::Vector{SpinBosonCoupling}    

    function SpinBosonSystem(N::Int)
        new(N, Float64[], zeros(Float64, N), SpinCoupling[], SpinBosonCoupling[])
    end
end

# --- API Mutators ---

"""
Adds a bosonic mode with frequency `w`. Returns the mode index.
"""
function add_boson!(sys::SpinBosonSystem, w::Float64)
    push!(sys.omega, w)
    return length(sys.omega)
end

"""
Sets the local splitting for a specific spin `i`.
"""
function set_epsilon!(sys::SpinBosonSystem, i::Int, val::Float64)
    sys.epsilon[i] = val
end

"""
Adds a two-body spin interaction at sites `i` and `j`.
"""
function add_spin_coupling!(sys::SpinBosonSystem, axis::Symbol, i::Int, j::Int, val::Float64)
    push!(sys.spin_couplings, SpinCoupling(axis, i, j, val))
end

"""
Adds a spin-boson interaction at site `i` and mode `m`.
"""
function add_spin_boson_coupling!(sys::SpinBosonSystem, m::Int, axis::Symbol, i::Int, val::Float64)
    push!(sys.spin_boson_couplings, SpinBosonCoupling(m, axis, i, val))
end

# --- Variational Manifolds ---

"""
Homogeneous Non-Gaussian Ansatz.
Contains the variational parameters.
"""
struct HomogeneousNGS <: AbstractAnsatz
    xi::Float64
    lambda::Float64
end

"""
Gaussian State (GS) Ansatz.
Structurally identical to HomogeneousNGS, 
but strictly locks parameters to zero.
"""
struct GS <: AbstractAnsatz
    xi::Float64
    lambda::Float64
    GS() = new(0.0, 0.0)
end

# --- Output State ---

"""
Output container for the ground state.
Carries the exact MPS and the optimized variational parameters.
"""
struct NGSState{A<:AbstractAnsatz}
    psi_spin::MPS
    var_params::A
end

"""
Telemetry data for the self-consistent optimization loop.
"""
struct SolverStats
    converged::Bool
    iterations::Int
    energy_history::Vector{Float64}
end