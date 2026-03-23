using Pkg
Pkg.activate(".")
using GeometricPCA
using LinearAlgebra
using Random
using Statistics
using CairoMakie
using Dates

# 1. Define a persistent workspace to hold all temporary memory
struct PCAWorkspace
    d_vec::Vector{Float64}
    v::Vector{Float64}
    w::Vector{Float64}
    q_vec::Vector{Float64}
    H::Matrix{Float64}
    Q_aug::Matrix{Float64}
    Q_new::Matrix{Float64}

    function PCAWorkspace(p::Int, d::Int)
        new(
            Vector{Float64}(undef, p),       # d_vec
            Vector{Float64}(undef, d),       # v
            Vector{Float64}(undef, p),       # w
            Vector{Float64}(undef, p),       # q_vec
            Matrix{Float64}(undef, d + 1, d),  # H
            Matrix{Float64}(undef, p, d + 1),  # Q_aug
            Matrix{Float64}(undef, p, d)     # Q_new
        )
    end
end

"""
Optimized, in-place adaptive principal subspace tracking.
"""
function update_subspace_optimized!(Q::Matrix{Float64}, R::Matrix{Float64}, a::Vector{Float64},
    n::Int, x::Vector{Float64}, W::PCAWorkspace)
    p, d = size(Q)
    inv_n1 = 1.0 / (n + 1)

    # 1. d_vec = x - a  (O(p) spatial deviation)
    @inbounds @simd for i in 1:p
        W.d_vec[i] = x[i] - a[i]
    end

    # 2. v = Q' * d_vec (O(pd) projection)
    mul!(W.v, Q', W.d_vec)

    # 3. Compute orthogonal residual: w = d - Q * v (O(pd))
    mul!(W.w, Q, W.v)
    rho_sq = 0.0
    @inbounds @simd for i in 1:p
        W.w[i] = W.d_vec[i] - W.w[i]
        rho_sq += W.w[i]^2
    end
    rho = sqrt(rho_sq)

    # 4. Normalize residual safely
    if rho > 1e-12
        inv_rho = 1.0 / rho
        @inbounds @simd for i in 1:p
            W.q_vec[i] = W.w[i] * inv_rho
        end
    else
        fill!(W.q_vec, 0.0)
    end

    # 5. Form (d+1) x d augmented core matrix H in-place
    @inbounds for j in 1:d
        for i in 1:d
            W.H[i, j] = R[i, j] + inv_n1 * W.v[i] * W.v[j]
        end
        W.H[d+1, j] = inv_n1 * rho * W.v[j]
    end

    # 6. Rank-1 QR update (Allocates slightly, but bounded purely by d^2, independent of p)
    F = qr!(W.H)
    Q_H = Matrix(F.Q)[:, 1:d]

    @inbounds for j in 1:d
        for i in 1:d
            R[i, j] = F.R[i, j]
        end
    end

    # 7. Update ambient basis: Q_new = [Q | q_vec] * Q_H (O(pd^2))
    # Construct augmented Q in-place
    @inbounds for j in 1:d
        @simd for i in 1:p
            W.Q_aug[i, j] = Q[i, j]
        end
    end
    @inbounds @simd for i in 1:p
        W.Q_aug[i, d+1] = W.q_vec[i]
    end

    # Multiply to get new basis
    mul!(W.Q_new, W.Q_aug, Q_H)

    # 8. Overwrite Q and update running mean (Welford's algorithm)
    copyto!(Q, W.Q_new)
    @inbounds @simd for i in 1:p
        a[i] += W.d_vec[i] * inv_n1
    end

    return n + 1
end


# High-dimensional system parameters
p = 200
d = 4
lambdas = [15.0, 10.0, 7.0, 5.0] # True spike eigenvalues
epsilon = 0.1                    # Background noise variance

total_stream_samples = 5e5
init_samples = 50                # Initial batch size for warm start

std_devs = sqrt.(vcat(lambdas, fill(epsilon, p - length(lambdas))))

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

workspace = PCAWorkspace(p, d)
record_interval = 10

num_records = Int(total_stream_samples) ÷ record_interval
ns = zeros(Int, num_records)
distances = zeros(Float64, num_records)
times = zeros(Float64, num_records)
allocs = zeros(Float64, num_records)

println("Commencing streaming PCA over $total_stream_samples new observations...")
x = Vector{Float64}(undef, p)

# Warm-up call to compile the function and prevent JIT overhead from skewing the metrics
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

filename = "figures/stream_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)
