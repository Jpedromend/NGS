# src/physics.jl

"""
Precompiles the ITensor OpSums to avoid redundancies.
Returns purely analytical operator sums; no tensor 
networks are instantiated here.
"""
function build_base_operators(model::SpinBosonSystem)
    N = model.N
    
    os_z = OpSum()
    for i in 1:N
        if abs(model.epsilon[i]) > 1e-14
            os_z += model.epsilon[i], "Sz", i
        end
    end
    
    os_x, os_x2 = OpSum(), OpSum()
    for i in 1:N
        os_x += 1.0, "Sx", i
        for j in 1:N
            os_x2 += 1.0, "Sx", i, "Sx", j
        end
    end
    
    os_Jx, os_Jy_y, os_Jy_z, os_Jz_z, os_Jz_y = OpSum(), OpSum(), OpSum(), OpSum(), OpSum()
    
    for c in model.spin_couplings
        if c.axis == :x
            os_Jx += c.val, "Sx", c.i, "Sx", c.j
        elseif c.axis == :y
            os_Jy_y += c.val, "Sy", c.i, "Sy", c.j
            os_Jy_z += c.val, "Sz", c.i, "Sz", c.j
        elseif c.axis == :z
            os_Jz_z += c.val, "Sz", c.i, "Sz", c.j
            os_Jz_y += c.val, "Sy", c.i, "Sy", c.j
        end
    end

    return (
        Hz = os_z, Hx = os_x, Hx2 = os_x2,
        HJx = os_Jx, HJy_y = os_Jy_y, HJy_z = os_Jy_z, HJz_z = os_Jz_z, HJz_y = os_Jz_y
    )
end

"""
Computes pre-contracted spin scalars and correlation matrices.
Evaluates transverse interaction energies strictly over the explicit coupling graph,
scaling linearly with the number of edges rather than N^2.
"""
function spin_averages(model::SpinBosonSystem, psi::MPS)
    exp_sz = expect(psi, "Sz")
    exp_sx = expect(psi, "Sx")
    
    avg_sz_tot = real(sum(exp_sz))
    avg_sx_tot = real(sum(exp_sx))
    
    C_xx = correlation_matrix(psi, "Sx", "Sx")
    C_yy = correlation_matrix(complex(psi), "Sy", "Sy")
    C_zz = correlation_matrix(psi, "Sz", "Sz")

    E_field = real(dot(model.epsilon, exp_sz))
    sx2_tot = real(sum(C_xx))

    A, B, C, D, E_xx = 0.0, 0.0, 0.0, 0.0, 0.0
    
    for coup in model.spin_couplings
        if coup.axis == :x
            E_xx += coup.val * real(C_xx[coup.i, coup.j])
        elseif coup.axis == :y
            A += coup.val * real(C_yy[coup.i, coup.j])
            B += coup.val * real(C_zz[coup.i, coup.j])
        elseif coup.axis == :z
            C += coup.val * real(C_yy[coup.i, coup.j])
            D += coup.val * real(C_zz[coup.i, coup.j])
        end
    end

    O_sum = A + B + C + D
    O_diff = A - B - C + D

    return (
        avg_sx_tot = avg_sx_tot,
        avg_sz_tot = avg_sz_tot,
        sx2_tot    = sx2_tot,
        E_field    = E_field,
        E_xx       = E_xx,
        O_sum      = O_sum,
        O_diff     = O_diff
    )
end

"""
Evaluates the analytical variational energy functional for the Homogeneous non-Gaussian state.
"""
function energy_cost(model::SpinBosonSystem, ansatz::HomogeneousNGS, obs)
    omega = model.omega[1]
    g = model.spin_boson_couplings[1].val 
    
    xi, lmd = ansatz.xi, ansatz.lambda
    
    K = (4.0 * g^2) / (model.N * omega)
    eta = (2.0 / model.N) * exp(-2.0 * xi) * (lmd * g / omega)^2

    E_boson = omega * sinh(xi)^2

    E_sb = obs.E_field * exp(-eta) -
           K * (1.0 - lmd)^2 * obs.avg_sx_tot^2 + 
           K * lmd * (lmd - 2.0) * obs.sx2_tot

    E_ss = - obs.E_xx - 0.5 * obs.O_sum - 0.5 * exp(-4.0 * eta) * obs.O_diff

    return E_boson + E_sb + E_ss
end

"""
Assembles the effective spin Hamiltonian by scaling the pre-built OpSums 
with the updated dressing parameters. The tensor network MPO is compiled 
strictly once per optimization step here.
"""
function effective_hamiltonian(base_ops, model::SpinBosonSystem, ansatz::HomogeneousNGS, avg_sx, sites; cutoff=1e-12)
    omega = model.omega[1]
    g = model.spin_boson_couplings[1].val 
    
    xi, lmd = ansatz.xi, ansatz.lambda

    K = (4.0 * g^2) / (model.N * omega)
    eta = (2.0 / model.N) * exp(-2.0 * xi) * (lmd * g / omega)^2

    c_field = exp(-eta)
    c_mf    = -2.0 * K * (1.0 - lmd)^2 * avg_sx
    c_quad  = K * lmd * (lmd - 2.0)
    
    c_plus  = 0.5 * (1.0 + exp(-4.0 * eta))
    c_minus = 0.5 * (1.0 - exp(-4.0 * eta))

    total_os = c_field * base_ops.Hz +
               c_mf * base_ops.Hx +
               c_quad * base_ops.Hx2 -
               1.0 * base_ops.HJx -
               c_plus * base_ops.HJy_y -
               c_minus * base_ops.HJy_z -
               c_plus * base_ops.HJz_z -
               c_minus * base_ops.HJz_y
            
    return MPO(total_os, sites; cutoff=cutoff)
end