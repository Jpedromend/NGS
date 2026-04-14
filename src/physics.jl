# src/physics.jl

"""
Computes pre-contracted spin scalars and correlation matrices 
required for the variational cost function and mean-field shifts.
"""
function spin_averages(model::ExtendedDickeModel, psi::MPS)
    exp_sz = expect(psi, "Sz")
    exp_sx = expect(psi, "Sx")
    
    avg_sz_tot = real(sum(exp_sz))
    avg_sx_tot = real(sum(exp_sx))
    
    C_xx = correlation_matrix(psi, "Sx", "Sx")
    C_yy = correlation_matrix(complex(psi), "Sy", "Sy")
    C_zz = correlation_matrix(psi, "Sz", "Sz")

    E_field = real(dot(model.epsilon, exp_sz))

    sx2_tot = real(sum(C_xx))

    # --- Important: Algebraic Factorization for Optimization ---
    # The solver evaluates the energy functional hundreds of times per step.
    # Re-contracting the tensor network (e.g. C_yy, C_zz) for every time 
    # would create a massive computational bottleneck.
    #
    # By algebraically factoring the eta-dependent terms out of the 
    # transverse energy expectations, we isolate the expensive contractions 
    # into static scalars (A, B, C, D). O_sum and O_diff group them, 
    # reducing the cost function (energy_cost) to trivial scalar arithmetic.
    
    A = dot(model.Jy, C_yy)
    B = dot(model.Jy, C_zz)
    C = dot(model.Jz, C_yy)
    D = dot(model.Jz, C_zz)

    E_xx = dot(model.Jx, C_xx)
    O_sum = A + B + C + D
    O_diff = A - B - C + D

    return (
        avg_sx_tot = avg_sx_tot,
        avg_sz_tot = avg_sz_tot,
        sx2_tot    = sx2_tot,
        E_field    = E_field,
        E_xx       = real(E_xx),
        O_sum      = real(O_sum),
        O_diff     = real(O_diff)
    )
end

"""
Evaluates the analytical variational energy functional.
"""
function energy_cost(model::ExtendedDickeModel, var_params, obs)
    xi, lmd = var_params
    
    K = (4.0*model.g^2)/(model.N*model.omega)
    eta = (2.0/model.N)*exp(-2.0*xi)*(lmd*model.g/model.omega)^2

    E_boson = model.omega * sinh(xi)^2

    E_sb = obs.E_field * exp(-eta) -
           K * (1.0 - lmd)^2 * obs.avg_sx_tot^2 + 
           K * lmd * (lmd - 2.0) * obs.sx2_tot

    # - Exx - 0.5 * O_sum - 0.5 * exp(-2eta) * O_diff
    E_ss = - obs.E_xx - 0.5 * obs.O_sum - 0.5 * exp(-4.0 * eta) * obs.O_diff

    return E_boson + E_sb + E_ss
end

"""
Precompiles the ITensor MPOs to avoid redundant matrix allocations 
during the self-consistent loop.
"""
function build_base_operators(model::ExtendedDickeModel, sites)
    N = model.N
    
    # Dressed atomic splitting
    os_z = OpSum()
    has_z = false
    for i in 1:N
        if abs(model.epsilon[i]) > 1e-14
            os_z += model.epsilon[i], "Sz", i
            has_z = true
        end
    end
    Hz = has_z ? MPO(os_z, sites) : nothing
    
    # Collective Sx terms (Always present for N > 0)
    os_x = OpSum()
    os_x2 = OpSum()
    for i in 1:N
        os_x += 1.0, "Sx", i
        for j in 1:N
            os_x2 += 1.0, "Sx", i, "Sx", j
        end
    end
    Hx = MPO(os_x, sites)
    Hx2 = MPO(os_x2, sites)
    
    # Dressed Spin-Spin Interactions (if any)
    rx, cx, vx = findnz(model.Jx)
    os_Jx = OpSum()
    for k in 1:length(vx) os_Jx += vx[k], "Sx", rx[k], "Sx", cx[k] end
    HJx = length(vx) > 0 ? MPO(os_Jx, sites) : nothing
    
    ry, cy, vy = findnz(model.Jy)
    os_Jy_y, os_Jy_z = OpSum(), OpSum()
    for k in 1:length(vy) 
        os_Jy_y += vy[k], "Sy", ry[k], "Sy", cy[k] 
        os_Jy_z += vy[k], "Sz", ry[k], "Sz", cy[k] 
    end
    HJy_y = length(vy) > 0 ? MPO(os_Jy_y, sites) : nothing
    HJy_z = length(vy) > 0 ? MPO(os_Jy_z, sites) : nothing
    
    rz, cz, vz = findnz(model.Jz)
    os_Jz_z, os_Jz_y = OpSum(), OpSum()
    for k in 1:length(vz) 
        os_Jz_z += vz[k], "Sz", rz[k], "Sz", cz[k] 
        os_Jz_y += vz[k], "Sy", rz[k], "Sy", cz[k] 
    end
    HJz_z = length(vz) > 0 ? MPO(os_Jz_z, sites) : nothing
    HJz_y = length(vz) > 0 ? MPO(os_Jz_y, sites) : nothing

    return (
        Hz = Hz, Hx = Hx, Hx2 = Hx2,
        HJx = HJx, HJy_y = HJy_y, HJy_z = HJy_z, HJz_z = HJz_z, HJz_y = HJz_y
    )
end

"""
Assembles the effective spin Hamiltonian by scaling `base_ops` 
with the updated dressing parameters.
"""
function effective_hamiltonian(base_ops, model::ExtendedDickeModel, var_params, avg_sx)
    xi, lmd = var_params

    K = (4.0 * model.g^2) / (model.N * model.omega)
    eta = (2.0 / model.N) * exp(-2.0 * xi) * (lmd * model.g / model.omega)^2

    c_field = exp(-eta)
    c_mf    = -2.0 * K * (1.0 - lmd)^2 * avg_sx
    c_quad  = K * lmd * (lmd - 2.0)
    
    c_plus  = 0.5 * (1.0 + exp(-4.0 * eta))
    c_minus = 0.5 * (1.0 - exp(-4.0 * eta))

    # Collect valid terms
    terms = MPO[]
    
    !isnothing(base_ops.Hz) && push!(terms, c_field * base_ops.Hz)
    
    push!(terms, c_mf * base_ops.Hx)
    push!(terms, c_quad * base_ops.Hx2)
    
    !isnothing(base_ops.HJx) && push!(terms, -1.0 * base_ops.HJx)
    
    if !isnothing(base_ops.HJy_y)
        push!(terms, -c_plus * base_ops.HJy_y)
        push!(terms, -c_minus * base_ops.HJy_z)
    end
    
    if !isnothing(base_ops.HJz_z)
        push!(terms, -c_plus * base_ops.HJz_z)
        push!(terms, -c_minus * base_ops.HJz_y)
    end
            
    return sum(terms)
end