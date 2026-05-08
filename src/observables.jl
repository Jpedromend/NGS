# src/observables.jl

"""
Computes expectation values. 
"""
function expect_ngs(op::String, state::NGSState, model::SpinBosonSystem)
    if op == "Sx"
        return expect(state.psi_spin, "Sx")
        
    elseif op == "Sz"
        g = model.spin_boson_couplings[1].val
        omega = model.omega[1]
        xi, lmd = state.var_params.xi, state.var_params.lambda
        
        eta = (2.0 / model.N) * exp(-2.0 * xi) * (lmd * g / omega)^2
        
        return exp(-eta) .* expect(state.psi_spin, "Sz")
        
    elseif op == "n"
        g = model.spin_boson_couplings[1].val
        omega = model.omega[1]
        xi, lmd = state.var_params.xi, state.var_params.lambda
        
        Komg = (4.0 * g^2) / (model.N * omega^2)
        
        sx_tot = real(sum(expect(state.psi_spin, "Sx")))
        sx2_tot = real(sum(correlation_matrix(state.psi_spin, "Sx", "Sx")))
        
        n_tot = sinh(xi)^2 + Komg * (1.0 - lmd^2) * (sx_tot^2) + Komg * lmd^2 * sx2_tot
        return n_tot / model.N
        
    else
        error("Observable '$op' is not implemented.")
    end
end

"""
Computes correlation matrices.
"""
function correlation_matrix_ngs(op1::String, op2::String, state::NGSState, model::SpinBosonSystem)
    if op1 == "Sx" && op2 == "Sx"
        return correlation_matrix(state.psi_spin, "Sx", "Sx")
        
    elseif op1 == "Sz" && op2 == "Sz"
        g = model.spin_boson_couplings[1].val
        omega = model.omega[1]
        xi, lmd = state.var_params.xi, state.var_params.lambda
        
        eta = (2.0 / model.N) * exp(-2.0 * xi) * (lmd * g / omega)^2
        
        corr_yy = correlation_matrix(complex(state.psi_spin), "Sy", "Sy")
        corr_zz = correlation_matrix(state.psi_spin, "Sz", "Sz")
        
        dressed_matrix = 0.5 .* corr_zz .* (1.0 + exp(-4.0 * eta)) .+
                         0.5 .* real.(corr_yy) .* (1.0 - exp(-4.0 * eta))
        return dressed_matrix
        
    else
        error("Correlation '$op1-$op2' is not implemented.")
    end
end