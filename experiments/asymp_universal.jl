using LinearAlgebra
using Random
using Statistics
using Distributions # Added for t-distribution
using Dates
using CairoMakie


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

# Helper functions for different distributions
"""
Fills matrix X with i.i.d. samples from {-1, 1} with probability 0.5 each.
Mean 0, Variance 1. Uses a thread-local RNG for robustness.
"""
function rand_bernoulli_pm1!(X::Matrix{Float64})
    # Use a thread-local RNG to ensure thread-safety and reproducibility within threads
    # The seed is based on threadid and a random integer to ensure uniqueness across runs/threads
    rng = MersenneTwister(Threads.threadid() + abs(rand(Int)))
    for i in eachindex(X)
        X[i] = rand(rng) < 0.5 ? -1.0 : 1.0
    end
    return X
end

"""
Fills matrix X with i.i.d. samples from a scaled t-distribution.
Mean 0, Variance 1, finite fourth moment (for df > 4). Uses a thread-local RNG for robustness.
"""
function rand_tdist_scaled!(X::Matrix{Float64}, df::Int=5)
    # Use a thread-local RNG to ensure thread-safety and reproducibility within threads
    rng = MersenneTwister(Threads.threadid() + abs(rand(Int)))
    # For TDist(df), mean is 0 (if df > 1), variance is df / (df - 2) (if df > 2)
    # For finite fourth moment, df > 4. Let's use df=5.
    # Variance for TDist(5) is 5 / (5 - 2) = 5/3.
    # To get variance 1, we need to scale by sqrt(1 / (5/3)) = sqrt(3/5).
    dist = TDist(df)
    scale_factor = sqrt((df - 2.0) / df) # To make variance 1
    for i in eachindex(X)
        X[i] = rand(rng, dist) * scale_factor
    end
    return X
end



# --------------------------------------------------------------------
#          Universality Test (LAPACK Version)
# --------------------------------------------------------------------
"""
Simulate the squared chordal distance between the true top-k subspace 
and the empirical top-k subspace over increasing sample sizes. Use incomplete eigendecomposition.
This version allows specifying the distribution for random matrix entries.
"""
function simulate_highdim_chordal_universal(ns::Vector{Int}, gamma::Float64, lambdas::Vector{Float64}, k::Int, n_trials::Int, dist_sampler!::Function)
    d = length(lambdas)
    total_runs = length(ns) * n_trials
    x_vals = zeros(Int, total_runs)
    y_vals = zeros(Float64, total_runs)

    sd_vals = sqrt.(lambdas)

    for (i, n) in enumerate(ns)
        p = round(Int, gamma * n)
        println("Simulating for n = $n, p = $p with distribution $(nameof(dist_sampler!))...")

        Threads.@threads :static for trial in 1:n_trials
            idx = (i - 1) * n_trials + trial

            X = Matrix{Float64}(undef, n, p)
            dist_sampler!(X) # Fill X with samples from the specified distribution

            # Scale the top d signal columns
            @inbounds for j in 1:d
                sd = sd_vals[j]
                @simd for row in 1:n
                    X[row, j] *= sd
                end
            end

            sq_norm = 0.0

            if p <= n
                A = X' * X
                vecs = eigen!(Symmetric(A), (p-k+1):p).vectors
                @inbounds for step in 1:k
                    @simd for row in 1:d
                        sq_norm += abs2(vecs[row, step])
                    end
                end
            else
                V = svd!(X).V
                @inbounds for step in 1:k
                    @simd for row in 1:d
                        sq_norm += abs2(V[row, step])
                    end
                end
            end

            x_vals[idx] = i
            y_vals[idx] = d + k - 2.0 * sq_norm
        end
    end

    return x_vals, y_vals
end


# let
ns = [40, 80, 160, 320] # Reduced for faster testing of universality
gammas = [0.5, 1.5]

lambdas = [6.0, 5.0, 4.0, 3.0, 2.0]
d = length(lambdas)
k = 3
n_trials = 200

# Note: randn! uses the global RNG, which is thread-safe in recent Julia versions.
# Our custom samplers now use thread-local RNGs.
dist_funcs = [randn!, rand_bernoulli_pm1!, rand_tdist_scaled!]
dist_labels = ["Gaussian", "Bernoulli (±1)", "t-dist (df=5)"]
dist_colors = [:blue, :green, :red]

fig = Figure(size=(1600, 550), fontsize=30)
Label(fig[0, 1:2], text="Universality of Random Matrices (d* = $d, k = $k, n_MC = $n_trials)",
    font=:bold)

axes = Axis[]

for (col_idx, gamma) in enumerate(gammas)
    ax = Axis(fig[1, col_idx],
        title="Macroscopic Bias (γ = $gamma)",
        xlabel="Sample Size (n)",
        ylabel=col_idx == 1 ? "Squared Chordal Distance" : "",
        xticks=(1:length(ns), string.(ns)),
        xticklabelrotation=pi / 4 # Rotate labels for better readability if many n's
    )
    push!(axes, ax)

    println("\n--- Starting Universality scenario: γ = $gamma ---")
    println("Computing theoretical macroscopic limit...")
    D_inf = macroscopic_limit(gamma, lambdas, k)
    D_inf_rounded = round(D_inf, digits=2)

    # Draw theoretical limit
    hlines!(ax, [D_inf], color=:darkorange, linestyle=:dash, linewidth=3, label="D∞(γ) = $D_inf_rounded")

    # Collect all y_vals for this gamma to set ylims later
    all_y_vals_for_gamma = Float64[]

    for (dist_idx, dist_sampler!) in enumerate(dist_funcs)
        x_vals_sim, y_vals_sim = simulate_highdim_chordal_universal(ns, gamma, lambdas, k, n_trials, dist_sampler!)

        # Adjust x_vals for plotting multiple boxplots per tick
        # Offset by -0.2, 0, 0.2 for 3 distributions
        x_vals_plot = x_vals_sim .+ (dist_idx - 2) * 0.2

        boxplot!(ax, x_vals_plot, y_vals_sim,
            color=(dist_colors[dist_idx], 0.6),
            width=0.15, # Smaller width for multiple boxes
            show_outliers=true,
            label=dist_labels[dist_idx])

        append!(all_y_vals_for_gamma, y_vals_sim)
    end
    axislegend(ax, position=:rt)

    # Set ylims based on collected data for this gamma, plus a buffer
    min_y = minimum(all_y_vals_for_gamma)
    max_y = maximum(all_y_vals_for_gamma)
    ylims!(ax, min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))
end

display(fig)

filename = "figures/universality_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)
# end