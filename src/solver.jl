# src/solver.jl

"""
Precompiles the initial tensor network environment (sites, operators, 
and state) to enable efficient warm-starting across parameter sweeps.
"""
function init_env(model::ExtendedDickeModel; maxdim=10)
    sites = siteinds("S=1/2", model.N)
    base_ops = build_base_operators(model, sites)
    psi_init = randomMPS(sites, maxdim)
    
    return (sites = sites, base_ops = base_ops, psi = psi_init)
end

"""
Dispatch traits for selecting the variational manifold.
"""
abstract type SolverManifold end
struct NGSDMRG <: SolverManifold end
struct GSDMRG  <: SolverManifold end

"""
Executes the self-consistent optimization loop. 
Defaults to NGS-DMRG (`NGS`).
"""
solve(model::ExtendedDickeModel; kwargs...) = solve(NGSDMRG(), model; kwargs...)

# Route NGSDMRG and GSDMRG to the unified internal engine with the correct flag
solve(::NGSDMRG, model::ExtendedDickeModel; kwargs...) = _solve(model; optimize_params=true, kwargs...)
solve(::GSDMRG, model::ExtendedDickeModel; kwargs...)  = _solve(model; optimize_params=false, kwargs...)

"""
Unified backend engine for the self-consistent protocol.
Alternates between DMRG sweeps for the exact spin sector and L-BFGS 
optimization for the bosonic dressing parameters.
"""
function _solve(model::ExtendedDickeModel;
                optimize_params=true,
                env=nothing,
                vparams_init=nothing,
                max_iter=250, 
                tol=1e-8,
                maxdims=[20,50,100,500], 
                nsweeps=500,
                etol = 1e-8,
                cutoff = [1e-10],
                noise = [1e-8,1e-8,1e-8,0.0],
                min_sweeps=20,
                outputlevel=1)

    # State and Operator Initialization
    if isnothing(env)
        sites = siteinds("S=1/2", model.N)
        psi = randomMPS(sites, maxdims[1])
        base_ops = build_base_operators(model, sites)
    else
        length(env.psi) == model.N || error("env.psi length != model.N")
        sites = env.sites
        psi = env.psi
        base_ops = env.base_ops
    end

    # Variational Parameters Initialization
    if optimize_params
        if isnothing(vparams_init)
            x_opt = [rand(-0.1:0.01:0.1), rand(0.0:0.1:0.5)]
        else
            x_opt = [vparams_init.xi, vparams_init.lambda]
        end
    else
        x_opt = [0.0, 0.0]
    end
    
    # Self-Consistent Loop
    spin_obs = spin_averages(model, psi)
    
    converged = false
    final_iter = max_iter
    energy_history = Vector{Float64}()
    
    prev_E0 = Inf
    E0 = energy_cost(model, x_opt, spin_obs)
    
    for iter in 1:max_iter
        if abs(E0-prev_E0) < tol
            if outputlevel == 1
                @printf("converged at step %3d\n", iter)
            end
            converged = true
            final_iter = iter
            break
        end
        if outputlevel == 1
            @printf("step %3d | E = %.8f | dE = %.2e | xi = %.4f | lam = %.4f\n", 
                iter, E0, abs(E0-prev_E0), x_opt[1], x_opt[2])
        end
        prev_E0 = E0

        # MPO Addition using the cached base_ops
        H_eff = effective_hamiltonian(base_ops, model, x_opt, spin_obs.avg_sx_tot)        
        
        # Run DMRG
        obs = DMRGObserver(energy_tol=etol, minsweeps=min_sweeps)
        _, psi = dmrg(H_eff, psi; nsweeps=nsweeps, maxdim=maxdims, cutoff=cutoff, noise=noise, observer=obs, outputlevel=outputlevel)
        
        # Optimize Variational Parameters
        spin_obs = spin_averages(model, psi)
        if optimize_params
            cost_fn(x) = energy_cost(model, x, spin_obs)
            res = optimize(cost_fn, x_opt, LBFGS(), Optim.Options(iterations=100))
            x_opt = Optim.minimizer(res)
            E0 = Optim.minimum(res)
        else
            x_opt = [0.0, 0.0]
            E0 = energy_cost(model, x_opt, spin_obs)
        end

        push!(energy_history, E0)
    end

    stats = SolverStats(
        converged,
        final_iter,
        energy_history
    )

    vparams = (xi = x_opt[1], lambda = x_opt[2])
    return SolverResult(E0, psi, vparams, stats)
    
end