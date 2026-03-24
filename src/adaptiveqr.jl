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


##############################
### Block Version ###########
##############################



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