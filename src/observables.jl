# src/observables.jl

"""
Computes the intensive mean photon number.
"""
function mean_photon_number(model::ExtendedDickeModel, res::SolverResult)
    xi, lmd = res.variational_params.xi, res.variational_params.lambda

    Komg = (4.0 * model.g^2) / (model.N * model.omega^2)
    obs = spin_averages(model, res.psi)

    n_tot = sinh(xi)^2 + Komg * (1.0 - lmd^2) * obs.avg_sx_tot^2 + Komg * lmd^2 * obs.sx2_tot
    
    return n_tot / model.N
end

"""
Evaluates the intensive collective magnetizations Mx and Mz.
"""
function collective_magnetizations(model::ExtendedDickeModel, res::SolverResult)
    xi, lmd = res.variational_params.xi, res.variational_params.lambda

    eta = (2.0 / model.N) * exp(-2.0 * xi) * (lmd * model.g / model.omega)^2
    obs = spin_averages(model, res.psi)

    mx = obs.avg_sx_tot / model.N
    mz = (exp(-eta) * obs.avg_sz_tot) / model.N

    return (Mx = mx, Mz = mz)
end

"""
Computes the longitudinal spin structure factor at a given momentum `k`.
"""
function structure_factor_z(model::ExtendedDickeModel, res::SolverResult, k::Real)
    xi, lmd = res.variational_params.xi, res.variational_params.lambda
    N = model.N

    corr_yy = correlation_matrix(complex(res.psi), "Sy", "Sy")
    corr_zz = correlation_matrix(res.psi, "Sz", "Sz")

    eta = (2.0 / N) * exp(-2.0 * xi) * (lmd * model.g / model.omega)^2

    val = 0.0 + 0.0im
    for n in 1:N, m in 1:N
        term = 0.5 * corr_zz[n,m] * (1.0 + exp(-4.0 * eta)) +
               0.5 * real(corr_yy[n,m]) * (1.0 - exp(-4.0 * eta))
        val += term * exp(im * k * (n - m))
    end

    return 4.0 * real(val) / (N^2)
end

"""
Evaluates the mixed atom-field correlators.
"""
function spin_boson_correlators(model::ExtendedDickeModel, res::SolverResult)
    xi, lmd = res.variational_params.xi, res.variational_params.lambda

    obs = spin_averages(model, res.psi)
    _sx = obs.avg_sx_tot
    _sz = obs.avg_sz_tot
    _sx2 = obs.sx2_tot
    
    _szx = real(sum(correlation_matrix(complex(res.psi), "Sz", "Sx")))

    varSx   = _sx2 - _sx^2
    covSzSx = _szx - _sz * _sx

    # dressing factors
    eta = (2.0 / model.N) * exp(-2.0 * xi) * (lmd * model.g / model.omega)^2
    a = (2.0 * sqrt(2.0) * model.g / (sqrt(model.N) * model.omega)) * lmd 
    
    return (C_Sx_x = -a * varSx, C_Sz_x = a * exp(-eta) * (0.5 * _sz - covSzSx))
end