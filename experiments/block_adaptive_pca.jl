using Pkg
Pkg.activate(".")
using GeometricPCA
using LinearAlgebra
using Random
using Statistics
using CairoMakie
using Dates

# 1. Define a persistent workspace to hold all temporary memory for block updates
struct BlockPCAWorkspace
    D::Matrix{Float64}
    V::Matrix{Float64}
    W::Matrix{Float64}
    Q_W::Matrix{Float64}
    R_W::Matrix{Float64}
    s::Vector{Float64}
    V_tilde::Matrix{Float64}
    H::Matrix{Float64}
    Q_aug::Matrix{Float64}
    Q_new::Matrix{Float64}
    Q_H::Matrix{Float64}

    function BlockPCAWorkspace(p::Int, d::Int, n0::Int)
        new(
            Matrix{Float64}(undef, p, n0),        # D
            Matrix{Float64}(undef, d, n0),        # V
            Matrix{Float64}(undef, p, n0),        # W
            Matrix{Float64}(undef, p, n0),        # Q_W
            Matrix{Float64}(undef, n0, n0),       # R_W
            Vector{Float64}(undef, d),            # s
            Matrix{Float64}(undef, d, n0),        # V_tilde
            Matrix{Float64}(undef, d + n0, d),    # H
            Matrix{Float64}(undef, p, d + n0),    # Q_aug
            Matrix{Float64}(undef, p, d),         # Q_new
            Matrix{Float64}(undef, d + n0, d)     # Q_H
        )
    end
end

"""
Optimized, in-place block-adaptive principal subspace tracking.
"""
function update_subspace_block!(Q::Matrix{Float64}, R::Matrix{Float64}, x_bar::Vector{Float64},
    n::Int, X_new::Matrix{Float64}, W::BlockPCAWorkspace)
    p, d = size(Q)
    _, n0 = size(X_new)

    # 1. D = X_new - x_bar * 1_{n_0}^T (O(p * n_0) spatial deviation)
    @inbounds for j in 1:n0
        @simd for i in 1:p
            W.D[i, j] = X_new[i, j] - x_bar[i]
        end
    end

    # 2. V = Q^T D (O(p * d * n_0) projection)
    mul!(W.V, Q', W.D)

    # 3. Compute orthogonal residual block W_mat = D - Q * V (O(p * d * n_0))
    mul!(W.W, Q, W.V)
    @inbounds for j in 1:n0
        @simd for i in 1:p
            W.W[i, j] = W.D[i, j] - W.W[i, j]
        end
    end

    # 4. Orthogonalize residual block: Q_W, R_W = EconomyQR(W) (O(p * n_0^2))
    F_W = qr!(W.W)
    fill!(W.Q_W, 0.0)
    @inbounds for i in 1:n0
        W.Q_W[i, i] = 1.0
    end
    # Apply Householder reflectors to extract the exact economy Q matrix
    lmul!(F_W.Q, W.Q_W)

    @inbounds for j in 1:n0
        for i in 1:n0
            W.R_W[i, j] = i <= j ? F_W.R[i, j] : 0.0
        end
    end

    # 5. s = V * 1_{n_0} (Row sums of projected deviations)
    fill!(W.s, 0.0)
    @inbounds for j in 1:n0
        @simd for i in 1:d
            W.s[i] += W.V[i, j]
        end
    end

    # 6. V_tilde = V - 1/(n + n_0) * s * 1_{n_0}^T (Implicit centering)
    inv_n_n0 = 1.0 / (n + n0)
    @inbounds for j in 1:n0
        @simd for i in 1:d
            W.V_tilde[i, j] = W.V[i, j] - inv_n_n0 * W.s[i]
        end
    end

    # 7. Form (d+n_0) x d augmented core matrix H in-place
    fill!(W.H, 0.0)
    @inbounds for j in 1:d
        @simd for i in 1:d
            W.H[i, j] = R[i, j]
        end
    end

    inv_n = 1.0 / n
    @inbounds for j in 1:d
        for i in 1:d
            sum_val = 0.0
            @simd for k in 1:n0
                sum_val += W.V[i, k] * W.V_tilde[j, k]
            end
            W.H[i, j] += inv_n * sum_val
        end
        for i in 1:n0
            sum_val = 0.0
            @simd for k in 1:n0
                sum_val += W.R_W[i, k] * W.V_tilde[j, k]
            end
            W.H[d+i, j] += inv_n * sum_val
        end
    end

    # 8. Dense QR update: Q_H, R = DenseQR(H)
    F_H = qr!(W.H)

    fill!(W.Q_H, 0.0)
    @inbounds for i in 1:d
        W.Q_H[i, i] = 1.0
    end
    lmul!(F_H.Q, W.Q_H)

    @inbounds for j in 1:d
        for i in 1:d
            R[i, j] = i <= j ? F_H.R[i, j] : 0.0
        end
    end

    # 9. Update ambient basis: Q = [Q | Q_W] * Q_H
    @inbounds for j in 1:d
        @simd for i in 1:p
            W.Q_aug[i, j] = Q[i, j]
        end
    end
    @inbounds for j in 1:n0
        @simd for i in 1:p
            W.Q_aug[i, d+j] = W.Q_W[i, j]
        end
    end

    mul!(W.Q_new, W.Q_aug, W.Q_H)
    copyto!(Q, W.Q_new)

    # 10. Update running mean: x_bar = x_bar + 1/(n + n_0) * D * 1_{n_0}
    @inbounds for j in 1:n0
        @simd for i in 1:p
            x_bar[i] += inv_n_n0 * W.D[i, j]
        end
    end

    # 11. Increment and return
    return n + n0
end


# High-dimensional system parameters
p = 200
d = 4
n0 = 10                          # Block (batch) size
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

filename = "figures/block_stream_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)