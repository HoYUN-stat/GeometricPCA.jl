# --------------------------------------------------------------------
#          Asymptotic behavior of Squared Chordal Distance for fixed p
# --------------------------------------------------------------------

# cd("experiments")
# ] activate .

using LinearAlgebra
using Random
using Statistics
using CairoMakie
using Dates

# --------------------------------------------------------------------
#                      Data Simulation Functions
# --------------------------------------------------------------------
"""
Simulate the empirical statistic for the covariance spiked model.
"""
function simulate_statistic(n::Int, p::Int, d::Int, ks::Vector{Int}, lambdas::Vector{Float64}, epsilon::Float64, n_trials::Int)
    # True covariance diagonal
    Sigma_diag = vcat(lambdas[1:d], fill(epsilon, p - d))
    std_devs = sqrt.(Sigma_diag)

    num_ks = length(ks)
    T_n_vals = zeros(n_trials, num_ks)

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

        for (idx, k) in enumerate(ks)
            # 4. Compute Frobenius norm squared of the intersection subspace
            sq_norm = 0.0
            @inbounds for j in (p-k+1):p
                @simd for i in 1:d
                    sq_norm += abs2(eig_decomp.vectors[i, j])
                end
            end

            # 5. Store final statistic: 2 * n * (k - sq_norm)
            @inbounds T_n_vals[trial, idx] = 2 * n * (k - sq_norm)
        end
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

# --------------------------------------------------------------------
#                      Configuration when k <= d
# --------------------------------------------------------------------
p = 30
d = 5
lambdas = [15.0, 12.0, 10.0, 8.0, 5.0] # True spike eigenvalues
epsilon = 1.0                          # Noise variance

ns = [10, 100, 1000, 10000] # Increasing sample sizes to observe convergence
n_trials = 2000

k1 = 3  # Case k <= d
k2 = 8  # Case k > d
ks = [k1, k2]

T_limit = generate_theoretical_limit(p, d, k1, lambdas, epsilon)

# Setup Figure using CairoMakie
fig = Figure(size=(800, 800), fontsize=25)
#Label(fig[0, 1], text="Asymptotic Behavior(p=$p, d* = $d, n_MC = $n_trials)",
    #font=:bold)

ax1 = Axis(fig[1, 1], title="Empirical Histograms (k = $k1 ≤ d*=$d)",
           xlabel=L"n [ l_{\Theta}(\mathcal{U}_{[d^{*}]}, \hat{\mathcal{V}}_{[k]})^{2} - (d^{*} - k) ]", ylabel="Density")
#ax2 = Axis(fig[1, 2], title="Empirical Histograms (k = $k2 > d*, Recentered)", xlabel="Statistic Value - 2*n*(k-d*)", ylabel="Density")

colors = [:red, :orange, :yellow, :green, :blue, :purple]

for (i, n) in enumerate(ns)
    println("Simulating for n = $n...")
    T_n_vals = simulate_statistic(n, p, d, ks, lambdas, epsilon, n_trials)

    # Left plot: k1
    density!(ax1, T_n_vals[:, 1], color=(colors[i], 0.4), label="n = $n", strokewidth=2)

    # Right plot: k2, recentered
    T_n_vals_k2 = T_n_vals[:, 2] .- 2 * n * (k2 - d)
    #density!(ax2, T_n_vals_k2, color=(colors[i], 0.4), label="n = $n", strokewidth=2)
end

# Add theoretical limit last on the first plot
density!(ax1, T_limit, color=:transparent, strokecolor=:black, strokewidth=3, linestyle=:dash, label="Theoretical")

axislegend(ax1, position=:rt)
xlims!(ax1, 0, 30)

#axislegend(ax2, position=:rt)

display(fig)

# Save the figure
filename = "figures/asymp_transition_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)
