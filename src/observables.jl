# src/observables.jl

"""
Computes the intensive mean photon number.
"""
function expect_n(model::ExtendedDickeModel, res::SolverResult)
    xi, lmd = res.variational_params.xi, res.variational_params.lambda
    Komg = (4.0 * model.g^2) / (model.N * model.omega^2)
    
    sx_tot = real(sum(expect_sx(model, res)))
    sx2_tot = real(sum(correlation_matrix_sxsx(model, res)))

    n_tot = sinh(xi)^2 + Komg * (1.0 - lmd^2) * (sx_tot^2) + Komg * lmd^2 * sx2_tot
    
    return n_tot / model.N
end

"""
Computes the site-resolved expectation value of Sx.
This operator commutes with the dressing unitary, requiring no transformation.
"""
function expect_sx(model::ExtendedDickeModel, res::SolverResult)
    return expect(res.psi, "Sx")
end

"""
Computes the dressed site-resolved expectation value of Sz.
"""
function expect_sz(model::ExtendedDickeModel, res::SolverResult)
    xi, lmd = res.variational_params.xi, res.variational_params.lambda
    N = model.N
    
    eta = (2.0 / N) * exp(-2.0 * xi) * (lmd * model.g / model.omega)^2
    
    return exp(-eta) .* expect(res.psi, "Sz")
end

"""
Computes the bare correlation matrix for SxSx.
This operator commutes with the dressing unitary.
"""
function correlation_matrix_sxsx(model::ExtendedDickeModel, res::SolverResult)
    return correlation_matrix(res.psi, "Sx", "Sx")
end

"""
Computes the fully dressed correlation matrix for SzSz, incorporating 
the hybrid spin-boson mixing with SySy correlations.
"""
function correlation_matrix_szsz(model::ExtendedDickeModel, res::SolverResult)
    xi, lmd = res.variational_params.xi, res.variational_params.lambda
    N = model.N

    eta = (2.0 / N) * exp(-2.0 * xi) * (lmd * model.g / model.omega)^2

    corr_yy = correlation_matrix(complex(res.psi), "Sy", "Sy")
    corr_zz = correlation_matrix(res.psi, "Sz", "Sz")

    dressed_matrix = 0.5 .* corr_zz .* (1.0 + exp(-4.0 * eta)) .+
                     0.5 .* real.(corr_yy) .* (1.0 - exp(-4.0 * eta))

    return dressed_matrix
end