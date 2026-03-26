# --------------------------------------------------------------------
#          Asymptotic behavior of Squared Chordal Distance for fixed γ
# --------------------------------------------------------------------

# cd("experiments")
# ] activate .
# dev ..
using LinearAlgebra
using Random
using Statistics
using KrylovKit
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

"""
Simulate the squared chordal distance between the true top-k subspace 
and the empirical top-k subspace over increasing sample sizes. Use incomplete eigendecomposition.
"""
function simulate_highdim_chordal(ns::Vector{Int}, gamma::Float64, lambdas::Vector{Float64}, k::Int, n_trials::Int)
    d = length(lambdas)
    total_runs = length(ns) * n_trials
    x_vals = zeros(Int, total_runs)
    y_vals = zeros(Float64, total_runs)

    sd_vals = sqrt.(lambdas)

    for (i, n) in enumerate(ns)
        p = round(Int, gamma * n)
        println("Simulating for n = $n, p = $p...")

        Threads.@threads :static for trial in 1:n_trials
            idx = (i - 1) * n_trials + trial

            # Local allocation to avoid threadid() bounds issues
            X = Matrix{Float64}(undef, n, p)
            randn!(X)

            # Scale the top d signal columns
            @inbounds for j in 1:d
                sd = sd_vals[j]
                @simd for row in 1:n
                    X[row, j] *= sd
                end
            end

            sq_norm = 0.0

            if p <= n
                # Path 1: gamma <= 1 (Small p). X'X is the smaller p x p matrix.
                A = X' * X

                # eigen! destroys A to save memory, extracting only top k
                vecs = eigen!(Symmetric(A), (p-k+1):p).vectors

                @inbounds for step in 1:k
                    @simd for row in 1:d
                        sq_norm += abs2(vecs[row, step])
                    end
                end
            else
                # Path 2: gamma > 1 (Large p). Let LAPACK handle the SVD natively.
                # svd! destroys X to save memory, which is fine since randn! overwrites it next loop
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

# --------------------------------------------------------------------
#          Combined Simulation (Old LAPACK Version) -> SEE BELOW
# --------------------------------------------------------------------
ns = [10, 20, 40, 80, 160, 320, 640, 1280, 2560] #WARNING!!!!: 2560 is really expensive! SEE BELOW
# ns = [10, 20, 40]
gammas = [0.5, 1.5] #WARNING: γ > 1 is really expensive!

lambdas = [6.0, 5.0, 4.0, 3.0, 2.0]    # True spike eigenvalues satisfying BBP
d = length(lambdas)
k = 3                        # Target empirical subspace dimension
n_trials = 200

fig = Figure(size=(900, 1000), fontsize=30)
Label(fig[0, 1], text="Asymptotic Behavior(d* = $d, k = $k, n_MC = $n_trials)",
    font=:bold)

for (row_idx, gamma) in enumerate(gammas)
    println("\n--- Starting scenario: γ = $gamma ---")
    println("Computing theoretical macroscopic limit...")
    D_inf = macroscopic_limit(gamma, lambdas, k)
    D_inf_rounded = round(D_inf, digits=2)

    x_vals, y_vals = simulate_highdim_chordal(ns, gamma, lambdas, k, n_trials)

    ax = Axis(fig[row_idx, 1],
        title="Macroscopic Bias (γ = $gamma)",
        xlabel="Sample Size (n)",
        ylabel="Squared Chordal Distance",
        xticks=(1:length(ns), string.(ns))
    )

    # Draw theoretical limit
    hlines!(ax, [D_inf], color=:darkorange, linestyle=:dash, linewidth=3, label="D∞(γ) = $D_inf_rounded")

    # Draw empirical distributions
    boxplot!(ax, x_vals, y_vals, color=(:royalblue, 0.6), width=0.4, show_outliers=true)

    axislegend(ax, position=:rt)
end

display(fig)

filename = "figures/macro_bias_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)

# --------------------------------------------------------------------
#          Post-Processing: Side-by-Side Layout
# --------------------------------------------------------------------
# Since the data is already computed and plotted in the `fig` object,
# we can simply rearrange the existing Makie axes without rerunning.

all_axes = filter(x -> x isa Axis, fig.content)
all_labels = filter(x -> x isa Label, fig.content)

if length(all_axes) >= 2
    println("Rearranging existing figure to side-by-side layout...")

    # Move the second axis to the first row, second column
    fig[1, 2] = all_axes[2]

    # Remove the overlapping Y-axis label on the right plot
    all_axes[2].ylabel = ""

    # Make the main title label span across both columns
    if !isempty(all_labels)
        fig[0, 1:2] = all_labels[1]
    end

    # Clean up the empty second row from the layout grid
    trim!(fig.layout)

    # Resize the figure to be wider and shorter to fix the vertical stretching
    resize!(fig, 1600, 550)

    # Restrict the Y-axis limits to cut out extreme outliers inflating the range
    ylims!(all_axes[1], 2.0, 3.5)
    ylims!(all_axes[2], 2.5, 5.2)

    display(fig)
end

filename_sbs = "figures/macro_bias_sidebyside_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename_sbs, fig)


# --------------------------------------------------------------------
#          Testing Incomplete Eigen Decomp
# --------------------------------------------------------------------
# using BenchmarkTools

# n, p = 2560, 1000  # Simulating one of your larger runs
# k = 3
# X = Matrix{Float64}(undef, n, p)
# randn!(X)
# lambdas = [6.0, 5.0, 4.0, 3.0, 2.0]
# sd_vals = sqrt.(lambdas)
# # Scale the top d signal columns
# @inbounds for j in 1:length(lambdas)
#     sd = sd_vals[j]
#     @simd for row in 1:n
#         X[row, j] *= sd
#     end
# end

# println("--- Accuracy Test ---")

# # LAPACK Full SVD
# lapack_svd = svd(X)
# lapack_top_k_rvecs = lapack_svd.V[:, 1:k]

# # KrylovKit Truncated SVD
# # We ask for k singular values. It returns the values, left/right vectors, and convergence info.
# vals, lvecs, rvecs, info = svdsolve(X, k, :LR);

# for i in 1:k
#     krylov_norm = norm(rvecs[i])
#     lapack_norm = norm(lapack_top_k_rvecs[:, i])
#     println("Vector $i - KrylovKit norm: $krylov_norm, LAPACK norm: $lapack_norm, diff: $(abs(krylov_norm - lapack_norm))")
# end

# println("\n--- Benchmark (Speed & Memory) ---")
# println("LAPACK top $k singular vectors:")
# @btime (s = svd($X); s.Vt[1:$k, :])  # Full SVD then slice

# println("\nKrylovKit top $k singular values:")
# @btime svdsolve($X, $k, :LR)



"""
Simulate the squared chordal distance using KrylovKit for truncated SVD.
"""
function simulate_highdim_chordal_krylov(ns::Vector{Int}, gamma::Float64, lambdas::Vector{Float64}, k::Int, n_trials::Int)
    d = length(lambdas)
    total_runs = length(ns) * n_trials
    x_vals = zeros(Int, total_runs)
    y_vals = zeros(Float64, total_runs)

    sd_vals = sqrt.(lambdas)

    for (i, n) in enumerate(ns)
        p = round(Int, gamma * n)
        println("Simulating for n = $n, p = $p...")

        Threads.@threads :static for trial in 1:n_trials
            idx = (i - 1) * n_trials + trial

            # Local allocation to avoid threadid() bounds issues
            X = Matrix{Float64}(undef, n, p)
            randn!(X)

            # Scale the top d signal columns
            @inbounds for j in 1:d
                sd = sd_vals[j]
                @simd for row in 1:n
                    X[row, j] *= sd
                end
            end

            # Use KrylovKit for fast truncated SVD
            # svdsolve returns a tuple: (singular values, left singular vectors, right singular vectors, info)
            _vals, _lvecs, rvecs, _info = svdsolve(X, k, :LR, tol=1e-10) # Added a stricter tolerance

            sq_norm = 0.0
            # rvecs is a vector of k vectors, each of length p
            @inbounds for vec in rvecs
                # We are interested in the projection onto the first d coordinates
                # This is safe because the simulation parameters ensure p >= d.
                @simd for row in 1:d
                    sq_norm += abs2(vec[row])
                end
            end

            x_vals[idx] = i
            y_vals[idx] = d + k - 2.0 * sq_norm
        end
    end

    return x_vals, y_vals
end

# --------------------------------------------------------------------
#          Combined Simulation (KrylovKit Version) -> HIGHLY UNSTABLE
# --------------------------------------------------------------------
# ns = [10, 20, 40, 80, 160, 320, 640, 1280, 2560] #WARNING: 2560 is really expensive!
ns = [10, 20, 40, 80, 160, 320, 640]
gammas = [0.5, 1.5] #WARNING: γ > 1 is really expensive!


lambdas = [6.0, 5.0, 4.0, 3.0, 2.0]    # True spike eigenvalues satisfying BBP
d = length(lambdas)
k = 3                        # Target empirical subspace dimension
n_trials = 200

# Create a figure with a side-by-side layout
fig = Figure(size=(1600, 550), fontsize=30)
Label(fig[0, 1:2], text="Asymptotic Behavior (Krylov, d* = $d, k = $k, n_MC = $n_trials)",
    font=:bold)

axes = Axis[]

for (col_idx, gamma) in enumerate(gammas)
    println("\n--- Starting KrylovKit scenario: γ = $gamma ---")
    println("Computing theoretical macroscopic limit...")
    D_inf = macroscopic_limit(gamma, lambdas, k)
    D_inf_rounded = round(D_inf, digits=2)

    x_vals, y_vals = simulate_highdim_chordal_krylov(ns, gamma, lambdas, k, n_trials)

    ax = Axis(fig[1, col_idx],
        title="Macroscopic Bias (γ = $gamma)",
        xlabel="Sample Size (n)",
        ylabel=col_idx == 1 ? "Squared Chordal Distance" : "",
        xticks=(1:length(ns), string.(ns))
    )
    push!(axes, ax)

    # Draw theoretical limit
    hlines!(ax, [D_inf], color=:darkorange, linestyle=:dash, linewidth=3, label="D∞(γ) = $D_inf_rounded")

    # Draw empirical distributions
    boxplot!(ax, x_vals, y_vals, color=(:royalblue, 0.6), width=0.4, show_outliers=true)

    axislegend(ax, position=:rt)
end

# Restrict the Y-axis limits to cut out extreme outliers
ylims!(axes[1], 2.0, 3.5)
ylims!(axes[2], 2.5, 5.2)

display(fig)

filename = "figures/macro_bias_krylov_sbs_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)
