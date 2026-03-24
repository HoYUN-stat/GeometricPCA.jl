# using GeometricPCA
using LinearAlgebra
using Random
using Statistics
using CairoMakie
using Dates

"""
Compute the theoretical deterministic macroscopic constant D_∞ 
based on the given high-dimensional limit theorem.
"""
function macroscopic_limit(gamma::Float64, lambdas::Vector{Float64}, k::Int)
    d = length(lambdas)
    sum_alignment = 0.0

    # Iterate over the intersection of I and J up to min(d, k)
    for j in 1:min(d, k)
        l = lambdas[j]

        # BBP condition check
        if l >= 1.0 + sqrt(gamma)
            num = 1.0 - gamma / (l - 1.0)^2
            den = 1.0 + gamma / (l - 1.0)
            sum_alignment += num / den
        else
            @warn "Eigenvalue λ = $l does not satisfy the BBP condition (λ >= $(1 + sqrt(gamma))) and is excluded!"
        end
    end

    return d + k - 2.0 * sum_alignment
end

"""
Simulate the squared chordal distance between the true top-k subspace 
and the empirical top-k subspace over increasing sample sizes. Use incomplete eigendecomposition.
"""
function simulate_highdim_chordal(ns::Vector{Int}, gamma::Float64, lambdas::Vector{Float64}, k::Int, n_trials::Int)
    d = length(lambdas)

    total_runs = length(ns) * n_trials
    x_vals = zeros(Int, total_runs)
    y_vals = zeros(Float64, total_runs)

    # Pre-calculate std deviations
    sd_vals = sqrt.(lambdas)

    for (i, n) in enumerate(ns)
        p = round(Int, gamma * n)
        println("Simulating for n = $n, p = $p...")

        Threads.@threads :static for trial in 1:n_trials
            idx = (i - 1) * n_trials + trial

            X = randn(n, p)

            # Scale the top d signal columns
            @inbounds for j in 1:d
                sd = sd_vals[j]
                @simd for row in 1:n
                    X[row, j] *= sd
                end
            end

            # Compute SVD: X = U * Diagonal(S) * Vt
            # svd(X).V gives the right singular vectors of X, corresponding to the eigenvectors of X'X.
            # The columns of V are naturally sorted in descending order of singular values.
            F = svd(X)

            sq_norm = 0.0

            # Direct projection: sum the squares of the first d rows of the top k empirical right singular vectors
            @inbounds for step in 1:k
                @simd for row in 1:d
                    sq_norm += abs2(F.V[row, step])
                end
            end

            x_vals[idx] = i
            y_vals[idx] = d + k - 2.0 * sq_norm
        end
    end

    return x_vals, y_vals
end

# Configuration setup mimicking (A1) and (A2)
gamma = 2.0                  # Limiting ratio p/n
lambdas = [6.0, 5.0, 4.0, 3.0, 2.0]    # True spike eigenvalues satisfying BBP
d = length(lambdas)
k = 3                        # Target empirical subspace dimension

ns = [10, 20, 40, 80, 160, 320, 640, 1280] # Increasing sample sizes
n_trials = 200

println("Computing theoretical macroscopic limit...")
D_inf = macroscopic_limit(gamma, lambdas, k)
D_inf_rounded = round(D_inf, digits=4)

x_vals, y_vals = simulate_highdim_chordal(ns, gamma, lambdas, k, n_trials)

# Plotting using CairoMakie boxplots
fig = Figure(size=(900, 600), fontsize=18)
ax = Axis(fig[1, 1],
    title="Macroscopic Bias (γ = $gamma, d = $d, k = $k, n_MC = $n_trials)",
    xlabel="Sample Size (n)",
    ylabel="Squared Chordal Distance",
    xticks=(1:length(ns), string.(ns))
)

# Draw theoretical limit
hlines!(ax, [D_inf], color=:darkorange, linestyle=:dash, linewidth=3, label="D_∞ = $D_inf_rounded")

# Draw empirical distributions
boxplot!(ax, x_vals, y_vals, color=(:royalblue, 0.6), width=0.4, show_outliers=true)

axislegend(ax, position=:rt)

display(fig)

filename = "figures/macro_bias_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)
