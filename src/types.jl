# src/types.jl

"""
Abstract supertype for all spin-boson physical models.
"""
abstract type AbstractSpinBosonModel end

"""
Helper structure for user-friendly model building. 
`axis` refers to the interaction axis (:x, :y, or :z).
"""
struct Couplings
    i::Int
    j::Int
    value::Float64
    axis::Symbol 
end

"""
Defines a single-mode extended Dicke model.

Contains the physical parameters: number of spins/sites `N`, cavity frequency `omega`,
collective coupling `g`, local fields `epsilon`, and sparse interaction matrices
`Jx`, `Jy`, `Jz` for anisotropic spin-spin couplings built from Couplings.
"""
struct ExtendedDickeModel <: AbstractSpinBosonModel
    N::Int
    omega::Float64
    g::Float64
    epsilon::Vector{Float64}
    Jx::SparseMatrixCSC{Float64, Int}
    Jy::SparseMatrixCSC{Float64, Int}
    Jz::SparseMatrixCSC{Float64, Int}

    function ExtendedDickeModel(N, omega, g, epsilon, Jx, Jy, Jz)
        new(N, Float64(omega), Float64(g), Float64.(epsilon), Jx, Jy, Jz)
    end
end

"""
Keyword constructor for `ExtendedDickeModel`. Expands a scalar `epsilon` to a uniform
vector of length `N` and constructs sparse coupling matrices from a list of `Couplings`.
"""
function ExtendedDickeModel(; N, omega, g, epsilon, couplings::AbstractVector{Couplings}=Couplings[])

    eps_vec = if epsilon isa AbstractVector
        length(epsilon) == N || error("epsilon must have length N=$N")
        Float64.(epsilon)
    else
        fill(Float64(epsilon), N)
    end

    Jx = spzeros(Float64, N, N)
    Jy = spzeros(Float64, N, N)
    Jz = spzeros(Float64, N, N)
    for c in couplings
        i, j = c.i, c.j

        (1 <= i <= N) || error("Coupling index i=$i out of bounds for N=$N")
        (1 <= j <= N) || error("Coupling index j=$j out of bounds for N=$N")
        i != j || error("Diagonal coupling not allowed: ($i,$j)")

        a = min(i, j)
        b = max(i, j)

        # upper-triangle convention to match internal Hamiltonian construction
        if c.axis == :x
            Jx[a, b] += c.value
        elseif c.axis == :y
            Jy[a, b] += c.value
        elseif c.axis == :z
            Jz[a, b] += c.value
        else
            error("Unknown coupling axis $(c.axis). Use :x, :y, or :z.")
        end
    end

    return ExtendedDickeModel(N, omega, g, eps_vec, Jx, Jy, Jz)
end

"""
Telemetry data for the self-consistent optimization loop.
"""
struct SolverStats
    converged::Bool
    iterations::Int
    energy_history::Vector{Float64}
end

"""
Container for the output of the variational solver.
"""
struct SolverResult{P}
    energy::Float64
    psi::MPS
    variational_params::P
    stats::SolverStats
end