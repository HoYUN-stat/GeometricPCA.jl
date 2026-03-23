using Pkg
Pkg.activate(".")
using GeometricPCA
using LinearAlgebra
using Random
using Statistics
using CairoMakie
using Dates

"""
Simulate the empirical statistic for the covariance spiked model.
"""
function simulate_statistic(n::Int, p::Int, d::Int, k::Int, lambdas::Vector{Float64}, epsilon::Float64, n_trials::Int)
    # True covariance diagonal
    Sigma_diag = vcat(lambdas[1:d], fill(epsilon, p - d))
    std_devs = sqrt.(Sigma_diag)

    T_n_vals = zeros(n_trials)

    # Allocate buffers exactly once to eliminate per-trial heap allocations
    X = Matrix{Float64}(undef, n, p)
    S = Matrix{Float64}(undef, p, p)

    for trial in 1:n_trials
        # 1. Generate standard normal data and scale by std_devs in-place
        randn!(X)
        @inbounds for j in 1:p
            sd = std_devs[j]
            @simd for i in 1:n
                X[i, j] *= sd
            end
        end

        # 2. Compute sample covariance: S = (X' * X) / n
        mul!(S, X', X)
        @inbounds @simd for i in eachindex(S)
            S[i] /= n
        end

        # 3. Eigen decomposition (Symmetric ensures real eigenvalues)
        eig_decomp = eigen!(Symmetric(S))

        # 4. Compute Frobenius norm squared of the intersection subspace
        sq_norm = 0.0
        @inbounds for j in (p-k+1):p
            @simd for i in 1:d
                sq_norm += abs2(eig_decomp.vectors[i, j])
            end
        end

        # 5. Store final statistic: 2 * n * (k - sq_norm)
        @inbounds T_n_vals[trial] = 2 * n * (k - sq_norm)
    end

    return T_n_vals
end
"""
Generate random samples from the theoretical asymptotic distribution.
"""
function generate_theoretical_limit(p::Int, d::Int, k::Int, lambdas::Vector{Float64}, epsilon::Float64, n_sims::Int=100_000)
    limit_vals = zeros(n_sims)

    # Precompute constants
    c_vals = [(lambdas[j] * epsilon) / (lambdas[j] - epsilon)^2 for j in 1:k]

    @inbounds for i in 1:n_sims
        total_val = 0.0
        for j in 1:k
            chi2_val = 0.0
            # Generate Chi-squared with p-d degrees of freedom directly on the fly
            @simd for _ in 1:(p-d)
                chi2_val += randn()^2
            end
            total_val += 2 * c_vals[j] * chi2_val
        end
        limit_vals[i] = total_val
    end

    return limit_vals
end

# Configuration
p = 30
d = 5
k = 3
lambdas = [15.0, 12.0, 10.0, 8.0, 5.0] # True spike eigenvalues
epsilon = 1.0                          # Noise variance

ns = [200, 500, 2000, 5000, 20000, 50000] # Increasing sample sizes to observe convergence
n_trials = 2000

println("Generating theoretical asymptotic distribution...")
T_limit = generate_theoretical_limit(p, d, k, lambdas, epsilon)

# Setup Figure using CairoMakie
fig = Figure(size=(1000, 800), fontsize=18)
Label(fig[0, 1:2], "Asymptotic Behavior (p = $p, d = $d, k = $k)", font=:bold, fontsize=20)

axs = Axis[]

for (i, n) in enumerate(ns)
    println("Simulating for n = $n...")
    T_n_vals = simulate_statistic(n, p, d, k, lambdas, epsilon, n_trials)

    row = (i - 1) ÷ 2 + 1
    col = (i - 1) % 2 + 1
    ax = Axis(fig[row, col], title="n = $n", xlabel="Statistic Value", ylabel="Density")
    push!(axs, ax)

    density!(ax, T_n_vals, color=(:royalblue, 0.5), label="Empirical", strokewidth=2)
    density!(ax, T_limit, color=(:darkorange, 0.5), label="Theoretical", strokewidth=2, linestyle=:dash)
    axislegend(ax, position=:rt)
end

linkaxes!(axs...)
for ax in axs
    xlims!(ax, 0, 30)
end

display(fig)

# Save the figure

filename = "figures/asymp_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)


######## Configuration when k > d
p = 30
d = 5
k = 7
lambdas = [15.0, 12.0, 10.0, 8.0, 5.0] # True spike eigenvalues
epsilon = 1.0                          # Noise variance

ns = [200, 500, 2000, 5000, 20000, 50000] # Increasing sample sizes to observe convergence
n_trials = 2000

# println("Generating theoretical asymptotic distribution...")
# T_limit = generate_theoretical_limit(p, d, k, lambdas, epsilon)

# Setup Figure using CairoMakie
fig = Figure(size=(1000, 800), fontsize=18)
Label(fig[0, 1:2], "Asymptotic Behavior (p = $p, d = $d, k = $k)", font=:bold, fontsize=20)

axs = Axis[]

for (i, n) in enumerate(ns)
    println("Simulating for n = $n...")
    T_n_vals = simulate_statistic(n, p, d, k, lambdas, epsilon, n_trials)

    # Recenter the diverging statistic by extracting the 4n drift
    T_n_vals .-= 4 * n

    row = (i - 1) ÷ 2 + 1
    col = (i - 1) % 2 + 1
    ax = Axis(fig[row, col], title="n = $n", xlabel="Statistic Value - 4 * n", ylabel="Density")
    push!(axs, ax)

    density!(ax, T_n_vals, color=(:royalblue, 0.5), label="Empirical", strokewidth=2)
    axislegend(ax, position=:rt)
end

# linkaxes!(axs...)
# for ax in axs
#     xlims!(ax, 0, 30)
# end

display(fig)

# Save the figure

filename = "figures/asymp_bloated_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)
