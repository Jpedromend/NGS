# NGS-DMRG: Variational non-Gaussian solutions to interacting spin-boson models

NGS-DMRG is a hybrid numerical framework for simulating the ground-state properties of strongly correlated spin-boson systems. 

Here we show the implementation of the method described in:
> <ins>Role of Matter Interactions in Superradiant Phenomena</ins>,
> JP Mendonça, K Jachymski, Y Wang,
> _Physical Review Letters_ 135 (**13**), 133601 (2025)

using the Dicke model as an example.

The necessary packages for this demo are:


```julia
using ITensors, ITensorMPS, Optim, LinearAlgebra
```

The Dicke model is given by
$$ 
\begin{aligned} 
H &= \omega a^\dagger a + \varepsilon \sum_j s^z_j + \frac{2g}{\sqrt{N}} \sum_j s^x_j (a + a^\dagger ) \\ 
&= \frac{\omega}{2} (x^2 + p^2) + \varepsilon S^z + g' S^x x 
\end{aligned} 
$$
where we have dropped constant energy shifts and defined the collective spin operators $S^\alpha = \sum_j s^\alpha_j$, the photon quadratures $x = (a^\dagger + a)/\sqrt{2}$ and $p = i(a^\dagger - a)/\sqrt{2}$, and the scaled coupling $g' = 2g/\sqrt{N/2}$.

and we choose the following model parameters


```julia
N = 20 # number of atoms
omega = 1.0 # cavity frequency
eps = 1.0 # atomic splitting
g = 0.8 # atom-cavity coupling strength
```

This framework combines:
1.  **Non-Gaussian Variational Ansatz:**

$$|\psi_{\rm NGS}\rangle = U_{\lambda} \left( U_{\mathrm{GS}} |0\rangle_{\rm b}\otimes|\phi\rangle_{\rm s} \right).$$

The bosonic sector is treated using a variational manifold combining displacement and squeezing, forming a Gaussian state. A (Lang-Firsov-inspired) polariton dressing transformation introduces entanglement between the two subsystems. This replaces explicit Hilbert space/photon number truncation.

In this notebook, we fix the above ansatz to a simplified form that has shown to be very efficient to (extended) Dicke models:
$$|\psi_{\rm NGS}\rangle = \exp\left(i \lambda \frac{g'}{\omega} S^x p\right) \exp(-i\Delta_x p) \exp\left(-\frac{i}{2}\xi (xp+px)\right) |0\rangle_{\rm b} \otimes |\phi\rangle_{\rm s} .$$

2.  **Tensor Network Solver:** The many-body spin state $|\phi\rangle_{\rm s}$ is obtained, with no further approximations, solving the effective spin Hamiltonian

$$H_{\rm eff} = \langle \psi_{\rm b} | U_\lambda^\dagger H U_\lambda | \psi_{\rm b} \rangle,$$

using Density Matrix Renormalization Group (DMRG) via `ITensors.jl` and `ITensorMPS.jl`, capturing spin-spin correlations.

3. **Self-Consistent optimization:**

A self-consistent optimization loop then minimizes the variational energy with respect to both the variational parameters and the spin wavefunction, ultimately combining the two steps above. 

An accurate approximation of the full spin-boson ground state beyond mean-field theory, i.e., beyond the factorized approximation, is obtained.

# Important Definitions


```julia
# Define the ITensor site types
sites = siteinds("S=1/2", N)
```

## Define Spin Operators as MPO


```julia
# Define the spin operators to define the effective Hamiltonian
sz_op = OpSum()
sx_op = OpSum()
sx2_op = OpSum()

for j in 1:N
    sx_op += "Sx", j
    sz_op += "Sz", j
    for k in 1:N
        sx2_op += "Sx", j, "Sx", k
    end
end

Sz = MPO(sz_op, sites)
Sx = MPO(sx_op, sites)
Sx2 = MPO(sx2_op, sites)
```

## Define the Energy Cost
The variational energy is exactly given by
$$
\mathcal{E} = \frac{\omega}{2} \Delta_x^2 + \omega \sinh^2(\xi) + \varepsilon \exp\left( - \frac{\lambda^2 g'^2}{4\omega^2} e^{-2\xi} \right) \langle S^z \rangle_{\rm s} + g'(1 - \lambda) \Delta_x \langle S^x \rangle_{\rm s} + \frac{g'^2}{2\omega} (\lambda^2 - 2\lambda) \langle (S^x)^2 \rangle_{\rm s} .
$$
where $\langle \cdot \rangle_{\rm s}$ represent lab-frame spin averages.
Moreover, in this example, the variational parameter $\Delta_x$ is minimized analytically:
$$ \Delta_x^* = -\frac{(1-\lambda)g'}{\omega} \langle S^x \rangle_{\rm s} , $$
such that
$$\mathcal{E} = \omega \sinh^2(\xi) + \varepsilon \exp\left( - \frac{2\lambda^2 g^2}{N\omega^2} e^{-2\xi} \right) \langle S^z \rangle_{\rm s} - \frac{4g^2}{N\omega} (1 - \lambda)^2 \langle S^x \rangle_{\rm s}^2 + \frac{4g^2}{N\omega} (\lambda^2 - 2\lambda) \langle (S^x)^2 \rangle_{\rm s} . $$

In code, we define ``K=(4.0*g^2)/(N*omega)`` and ``eta = (2.0/N)*exp(-2.0*xi)*(lmd*g/omega)^2`` to simplify the equations.


```julia
# Define the variational energy cost to be minimized
function energy_cost(model_params, var_params, spin_avgs)
    omega, eps, g, N = model_params
    xi, lmd = var_params
    sz, sx, sx2 = spin_avgs

    K = (4.0*g^2)/(N*omega)
    eta = (2.0/N)*exp(-2.0*xi)*(lmd*g/omega)^2

    return omega*sinh(xi)^2 + eps*exp(-eta)*sz - K*( (1-lmd)*sx )^2 + K*(lmd^2 - 2.0*lmd)*sx2
end
```


```julia
# Define the spin averages
function spin_averages(psi)
    avg_sx = sum(expect(psi, "Sx"))
    avg_sz = sum(expect(psi, "Sz"))
    corr_sxsx = correlation_matrix(psi, "Sx", "Sx")
    avg_sx2 = sum(corr_sxsx)

    return real(avg_sz), real(avg_sx), real(avg_sx2)
end
```

## Define the Effective Spin Hamiltonian

The effective Hamiltonian (fixing $\Delta_x^*$ and constants dropped):
$$H_{\rm eff} = \varepsilon \exp\left( - \frac{2\lambda^2 g^2}{N\omega^2} e^{-2\xi} \right) S^z - \frac{8g^2}{N\omega} (1 - \lambda)^2 \langle S^x \rangle S^x + \frac{4g^2}{N\omega} (\lambda^2 - 2\lambda) (S^x)^2$$

## The Self-Consistent Loop


```julia
# State Initialization
psi = randomMPS(sites, 10)
xi, lmd = [rand(-0.1:0.01:0.1), rand(0.0:0.1:0.5)]
_sz,_sx,_sx2 = spin_averages(psi)
```


```julia
K = (4.0*g^2)/(N*omega)
# Loop Initialization
max_iter = 500
prev_E0 = Inf
E0 = 0.0
for iter in 1:max_iter
    # check convergence
    if abs(E0-prev_E0) < 1e-8
        println("converged at step $iter")
        break
    end
    println("step $iter: energy difference = $(abs(E0-prev_E0))")
    prev_E0 = E0
    
    # Built the effective spin Hamiltonian
    eta = (2.0/N)*exp(-2.0*xi)*(lmd*g/omega)^2
    H_eff = eps*exp(-eta)*Sz - 2.0*K*((1-lmd)^2)*_sx*Sx + K*(lmd^2 - 2.0*lmd)*Sx2
    
    # Run DMRG
    obs = DMRGObserver(energy_tol=1e-8, minsweeps=20)
    _, psi = dmrg(H_eff, psi; nsweeps=500, maxdim=[20,50,100,250], cutoff=[1e-8], observer=obs, outputlevel=1)

    # Update spin averages
    _sz,_sx,_sx2 = spin_averages(psi)
    
    # Optimize Variational Parameters
    cost_func(x) = energy_cost([omega, eps, g, N], x, [_sz,_sx,_sx2])
    res = optimize(cost_func, [xi,lmd], LBFGS(), Optim.Options(iterations=100))
    xi,lmd = Optim.minimizer(res)
    E0 = Optim.minimum(res)
end
```

## Final Converged Result

The explicit equations for the expectation values in the non-Gaussian ansatz are:

$$\langle n \rangle_{\rm NGS} = \sinh^2(\xi) + \frac{4g^2}{N\omega^2} (1-\lambda^2) \langle S^x \rangle^2 + \frac{4 g^2}{N\omega^2} \lambda^2 \langle (S^x)^2 \rangle$$

$$\langle S^z \rangle_{\rm NGS} = \exp\left( - \frac{2\lambda^2 g^2}{N\omega^2} e^{-2\xi} \right) \langle S^z \rangle$$


```julia
# Update the coefficients and bare averages
eta = (2.0/N)*exp(-2.0*xi)*(lmd*g/omega)^2
Komg = (4.0*g^2)/(N*omega^2)
_sz,_sx,_sx2 = spin_averages(psi)

# Observables
avg_n = sinh(xi)^2 + Komg*(1-lmd^2)*_sx^2 + Komg*_sx2*lmd^2
avg_sz = exp(-eta)*_sz

println("Mean photon number: $(avg_n/N) \nMean magnetization: $(avg_sz/N)")
```

# Quantum Phase Transitions

We now loop over $g$ to explore the QPT.


```julia
# Model Parameters
N = 100 # number of atoms
omega = 1.0 # cavity frequency
eps = 1.0 # atomic splitting

# Define the ITensor site types
sites = siteinds("S=1/2", N)

# Define the spin operators to define the effective Hamiltonian
sz_op = OpSum()
sx_op = OpSum()
sx2_op = OpSum()

for j in 1:N
    sx_op += "Sx", j
    sz_op += "Sz", j
    for k in 1:N
        sx2_op += "Sx", j, "Sx", k
    end
end

Sz = MPO(sz_op, sites)
Sx = MPO(sx_op, sites)
Sx2 = MPO(sx2_op, sites)
```


```julia
# State Initialization
psi = randomMPS(sites, 10)
xi, lmd = [rand(-0.1:0.01:0.1), rand(0.0:0.1:0.5)]
_sz,_sx,_sx2 = spin_averages(psi)
```


```julia
# Loop in g
g_list = range(0, 1.0, length=20)
E0_list = Float64[]
avg_n_list = Float64[]
avg_sz_list = Float64[]
for g in g_list
    K = (4.0*g^2)/(N*omega)
    # Loop Initialization
    max_iter = 500
    prev_E0 = Inf
    E0 = 0.0
    for iter in 1:max_iter
        # check convergence
        if abs(E0-prev_E0) < 1e-8
            println("for g=$g, converged at step $iter")
            break
        end
        println("step $iter: energy difference = $(abs(E0-prev_E0))")
        prev_E0 = E0
        
        # Built the effective spin Hamiltonian
        eta = (2.0/N)*exp(-2.0*xi)*(lmd*g/omega)^2
        H_eff = eps*exp(-eta)*Sz - 2.0*K*((1-lmd)^2)*_sx*Sx + K*(lmd^2 - 2.0*lmd)*Sx2
        
        # Run DMRG
        obs = DMRGObserver(energy_tol=1e-8, minsweeps=20)
        _, psi = dmrg(H_eff, psi; nsweeps=500, maxdim=[20,50,100,250], cutoff=[1e-8], observer=obs, outputlevel=1)
    
        # Update spin averages
        _sz,_sx,_sx2 = spin_averages(psi)
        
        # Optimize Variational Parameters
        cost_func(x) = energy_cost([omega, eps, g, N], x, [_sz,_sx,_sx2])
        res = optimize(cost_func, [xi,lmd], LBFGS(), Optim.Options(iterations=100))
        xi,lmd = Optim.minimizer(res)
        E0 = Optim.minimum(res)
    end
    # Save converged observables
    eta = (2.0/N)*exp(-2.0*xi)*(lmd*g/omega)^2
    Komg = (4.0*g^2)/(N*omega^2)
    _sz,_sx,_sx2 = spin_averages(psi)
    
    # Observables
    avg_n = sinh(xi)^2 + Komg*(1-lmd^2)*_sx^2 + Komg*_sx2*lmd^2
    avg_sz = exp(-eta)*_sz

    push!(E0_list, E0/N)
    push!(avg_n_list, avg_n/N)
    push!(avg_sz_list, avg_sz/N)
end
```


```julia
using PyPlot
using LaTeXStrings
```


```julia
fig, ax = subplots(1, 3, figsize=(14, 4), sharex=true)

ax[1].plot(g_list, E0_list, marker="o", markersize=4, color="tab:brown")
ax[2].plot(g_list, avg_n_list, marker="o", markersize=4, color="tab:blue")
ax[3].plot(g_list, avg_sz_list, marker="o", markersize=4, color="tab:red")

ax[1].set_ylabel(L"$E_0/N$")
ax[2].set_ylabel(L"$\langle n \rangle/N$")
ax[3].set_ylabel(L"$\langle S^z \rangle/N$")

for axi in ax
    axi.set_xlabel(L"$g$")
    axi.axvline(x=0.5, ls="--", color="gray", alpha=0.6)
end

ax[1].set_xlim(0, 1)
ax[2].set_ylim(bottom=0)
ax[3].set_ylim(-0.5, 0)

fig.tight_layout()
```
