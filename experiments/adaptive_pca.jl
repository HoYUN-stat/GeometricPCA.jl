# cd("experiments")
# ] activate .
# dev ..
using GeometricPCA
using LinearAlgebra
using Random
using Statistics
using CairoMakie
using Dates

# --------------------------------------------------------------------
#          Parameter Setup
# --------------------------------------------------------------------
# High-dimensional system parameters
p = 200
lambdas = [15.0, 10.0, 7.0, 5.0] # True spike eigenvalues
d = length(lambdas)
noise_var = 0.1                    # Background noise variance
std_devs = sqrt.(vcat(lambdas, fill(noise_var, p - d)))

total_stream_samples = 5e5

# --------------------------------------------------------------------
#          Init for Warm Start
# --------------------------------------------------------------------

init_samples = 50                # Initial batch size for warm start
X_init = randn(init_samples, p) .* std_devs'
a = vec(mean(X_init, dims=1))
X_cen = X_init .- a'
A_unscaled = X_cen' * X_cen

F_init = eigen(Symmetric(A_unscaled))
Q = F_init.vectors[:, end:-1:end-d+1]
R = Matrix(Diagonal(F_init.values[end:-1:end-d+1]))

# --------------------------------------------------------------------
#          Rank-1 QR Update
# --------------------------------------------------------------------
n = init_samples
workspace = PCAWorkspace(p, d)
record_interval = 10

num_records = Int(total_stream_samples) ÷ record_interval
ns = zeros(Int, num_records)
distances = zeros(Float64, num_records)
times = zeros(Float64, num_records)
allocs = zeros(Float64, num_records)

println("Commencing streaming PCA over $total_stream_samples new observations...")
x = Vector{Float64}(undef, p)
randn!(x);
x .*= std_devs;
update_subspace_optimized!(copy(Q), copy(R), copy(a), n, x, workspace)

total_time = 0.0
total_alloc_bytes = 0

for i in 1:Int(total_stream_samples)
    randn!(x)
    x .*= std_devs

    mem_before = Base.gc_bytes()
    t_before = time_ns()

    n = update_subspace_optimized!(Q, R, a, n, x, workspace)

    t_after = time_ns()
    mem_after = Base.gc_bytes()

    total_time += (t_after - t_before) / 1e9
    total_alloc_bytes += (mem_after - mem_before)

    if i % record_interval == 0
        sq_norm = sum(abs2, view(Q, 1:d, :))
        dist = 2.0 * (d - sq_norm)

        idx = i ÷ record_interval
        ns[idx] = n
        distances[idx] = dist
        times[idx] = total_time
        allocs[idx] = total_alloc_bytes / (1024^2) # Convert bytes to MB
    end
end
println("Streaming complete.")

# Draw Performance and Convergence Plots
fig = Figure(size=(1400, 450), fontsize=18)

ax1 = Axis(fig[1, 1],
    title="Subspace Convergence (p = $p, d = $d)",
    xlabel="Sequential samples (n)",
    ylabel="Squared Chordal Distance"
)
lines!(ax1, ns, distances, color=:royalblue, linewidth=2)

ax2 = Axis(fig[1, 2],
    title="Cumulative Compute Time",
    xlabel="Sequential samples (n)",
    ylabel="Time (Seconds)"
)
lines!(ax2, ns, times, color=:darkorange, linewidth=2)

ax3 = Axis(fig[1, 3],
    title="Cumulative Memory Allocated",
    xlabel="Sequential samples (n)",
    ylabel="Memory (MB)"
)
lines!(ax3, ns, allocs, color=:forestgreen, linewidth=2)

display(fig)

# filename = "figures/stream_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
# save(filename, fig)


# --------------------------------------------------------------------
#          Block Version
# --------------------------------------------------------------------
n0 = 10                          # Block (batch) size

println("Initializing subspace tracking with $init_samples warm start samples...")
X_init = randn(init_samples, p) .* std_devs'
a = vec(mean(X_init, dims=1))

# Compute exact unscaled scatter matrix for initialization
X_cen = X_init .- a'
A_unscaled = X_cen' * X_cen

F_init = eigen(Symmetric(A_unscaled))
Q = F_init.vectors[:, end:-1:end-d+1]
R = Matrix(Diagonal(F_init.values[end:-1:end-d+1]))
n = init_samples

workspace = BlockPCAWorkspace(p, d, n0)
record_interval = 1 # Record stats every batch (n0 samples)

num_batches = Int(total_stream_samples) ÷ n0
num_records = num_batches ÷ record_interval
ns = zeros(Int, num_records)
distances = zeros(Float64, num_records)
times = zeros(Float64, num_records)
allocs = zeros(Float64, num_records)

println("Commencing block-streaming PCA over $total_stream_samples new observations in blocks of $n0...")
X_new = Matrix{Float64}(undef, p, n0)

# Warm-up call to compile the function and prevent JIT overhead from skewing the metrics
randn!(X_new);
X_new .*= std_devs;
update_subspace_block!(copy(Q), copy(R), copy(a), n, X_new, workspace)

total_time = 0.0
total_alloc_bytes = 0

for i in 1:num_batches
    randn!(X_new)
    X_new .*= std_devs

    mem_before = Base.gc_bytes()
    t_before = time_ns()

    n = update_subspace_block!(Q, R, a, n, X_new, workspace)

    t_after = time_ns()
    mem_after = Base.gc_bytes()

    total_time += (t_after - t_before) / 1e9
    total_alloc_bytes += (mem_after - mem_before)

    if i % record_interval == 0
        sq_norm = sum(abs2, view(Q, 1:d, :))
        dist = 2.0 * (d - sq_norm)

        idx = i ÷ record_interval
        ns[idx] = n
        distances[idx] = dist
        times[idx] = total_time
        allocs[idx] = total_alloc_bytes / (1024^2) # Convert bytes to MB
    end
end
println("Streaming complete.")

# Setup Makie Layout and Draw the plots
fig = Figure(size=(1400, 450), fontsize=18)

ax1 = Axis(fig[1, 1], title="Block Subspace Convergence (p = $p, d = $d, n₀ = $n0)", xlabel="Sequential samples (n)", ylabel="Squared Chordal Distance")
lines!(ax1, ns, distances, color=:royalblue, linewidth=2)

ax2 = Axis(fig[1, 2], title="Cumulative Compute Time", xlabel="Sequential samples (n)", ylabel="Time (Seconds)")
lines!(ax2, ns, times, color=:darkorange, linewidth=2)

ax3 = Axis(fig[1, 3], title="Cumulative Memory Allocated", xlabel="Sequential samples (n)", ylabel="Memory (MB)")
lines!(ax3, ns, allocs, color=:forestgreen, linewidth=2)

display(fig)

# filename = "figures/block_stream_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
# save(filename, fig)