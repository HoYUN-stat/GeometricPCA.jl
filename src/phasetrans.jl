# ------------------------------------------------------------------
# 1. Continuous Alignment Function
# ------------------------------------------------------------------
"""
    weight_bbp(s::Float64, γ::Float64; α::Float64=1.0) -> Float64

Computes the asymptotic cross-alignment limit `δ_l=f(s_l)`.
Handles the ideal (α=1.0) and data-split (0 < α < 1) cases.
"""
function weight_bbp(s::Float64, γ::Float64; α::Float64=1.0)
    @assert γ > 0.0 "Gamma must be strictly positive."
    @assert 0.0 < α <= 1.0 "Split ratio α must be in (0, 1]."

    if α == 1.0
        # Full-sample ideal alignment
        @assert s > 1.0 "Signal must be strictly super-critical (s > 1)."
        return (1.0 - 1.0 / s^2) / (1.0 + sqrt(γ) / s)
    else
        # Data-splitting cross-alignment
        s_min = 1.0 / sqrt(min(α, 1.0 - α))
        @assert s > s_min "Signal must survive both splits (s > max(1/sqrt(α), 1/sqrt(1-α)))."

        term1 = (1.0 - 1.0 / (α * s^2)) / (1.0 + sqrt(γ) / (α * s))
        term2 = (1.0 - 1.0 / ((1.0 - α) * s^2)) / (1.0 + sqrt(γ) / ((1.0 - α) * s))
        return term1 * term2
    end
end


# ------------------------------------------------------------------
# 2. Strict BBP Detectability Index (d_+)
# ------------------------------------------------------------------
"""
    upper_idx(lambdas::Vector{Float64}, γ::Float64; α::Float64=1.0) -> Int

Returns `d_+(γ)` or `d_+(γ1, γ2)`, the number of spikes that strictly survive 
the BBP transition in all available data splits.
"""
function upper_idx(lambdas::Vector{Float64}, γ::Float64; α::Float64=1.0)
    @assert γ > 0.0
    @assert 0.0 < α <= 1.0

    # Determine the strict eigenvalue threshold based on the split
    if α == 1.0
        λ_thresh = 1.0 + sqrt(γ)
    else
        # The threshold is dictated by the smaller data split (which has the larger effective γ)
        γ_max = γ / min(α, 1.0 - α)
        λ_thresh = 1.0 + sqrt(γ_max)
    end

    # Find the last index strictly greater than the threshold
    d_plus = 0
    @inbounds for l in eachindex(lambdas)
        if lambdas[l] > λ_thresh
            d_plus = l
        else
            break
        end
    end

    return d_plus
end


# ------------------------------------------------------------------
# 3. Critical Signal Threshold Finder (s*)
# ------------------------------------------------------------------
"""
    signal_thres(γ::Float64; α::Float64=1.0)

Returns the critical signal strength s_*(γ, α) where the alignment exceeds 1/2.
Uses closed-form solutions for α=1.0 and α=0.5, and an allocation-free bisection 
for arbitrary α.
"""
function signal_thres(γ::Float64; α::Float64=1.0)
    @assert γ > 0.0
    @assert 0.0 < α <= 1.0

    if α == 1.0
        return (sqrt(γ) + sqrt(γ + 8.0)) / 2.0

    elseif α == 0.5
        return (sqrt(2.0) + 1.0) * (sqrt(γ) + sqrt(γ + 4.0 - 2.0 * sqrt(2.0)))

    else
        # The function f_α(s) is strictly increasing, so bisection is guaranteed to converge.
        s_min = 1.0 / sqrt(min(α, 1.0 - α))

        # 1. Bracket the root
        s_low = s_min + 1e-6
        s_high = s_low + 1.0
        while weight_bbp(s_high, γ; α=α) < 0.5
            s_high *= 2.0
        end

        # 2. Bisect to machine precision
        tol = 1e-12
        for _ in 1:100
            s_mid = 0.5 * (s_low + s_high)
            val = weight_bbp(s_mid, γ; α=α)

            if abs(val - 0.5) < tol
                return s_mid
            elseif val < 0.5
                s_low = s_mid
            else
                s_high = s_mid
            end
        end
        return 0.5 * (s_low + s_high)
    end
end



# ------------------------------------------------------------------
# 4. Macroscopic Geometric Risk / Subspace Recovery
# ------------------------------------------------------------------
"""
    limit_geometric_risk(lambdas::Vector{Float64}, γ::Float64, D::Int, k::Int; α::Float64=1.0) -> Float64

Computes the deterministic limit of the chordal distance limit between the reference 
subspace (dimension D) and the validation subspace (dimension k).
Handles both ideal (α=1.0) and data-splitting (0 < α < 1) limits.
"""
function limit_geometric_risk(lambdas::Vector{Float64}, γ::Float64, D::Int, k::Int; α::Float64=1.0)
    @assert γ > 0.0 "Gamma must be strictly positive."
    @assert 0.0 < α <= 1.0 "Split ratio α must be in (0, 1]."
    @assert D >= 1 && k >= 1 "Dimensions must be strictly positive."

    # Find the number of strictly BBP-detectable spikes
    d_plus = upper_idx(lambdas, γ; α=α)

    # The sum is evaluated strictly over the intersection (I ∩ J)_γ
    # which corresponds to indices up to min(|I|, |J|, d_plus)
    max_idx = min(D, k, d_plus)

    sum_alignment = 0.0

    # Enforce zero-allocation, vectorized loop for the summation
    @inbounds @simd for l in 1:max_idx
        # Extract signal strength s_l from the eigenvalue parameterization
        s_l = (lambdas[l] - 1.0) / sqrt(γ)

        # weight_bbp inherently evaluates the appropriate cross-alignment 
        # based on the splitting ratio α
        sum_alignment += weight_bbp(s_l, γ; α=α)
    end

    # D_∞ = (|I| + |J|) - 2 * sum(alignments)
    return Float64(D + k) - 2.0 * sum_alignment
end

# ------------------------------------------------------------------
# 5. Optimal Truncation Index (d_*)
# ------------------------------------------------------------------
"""
    lower_idx(lambdas::Vector{Float64}, γ::Float64; α::Float64=1.0) -> Int

Returns `d_*(γ)` or `d_*(γ1, γ2)`, the optimal truncation index.
This is the maximum index `l` such that the asymptotic alignment limit `f(s_l) > 1/2`,
meaning the signal is strong enough to actively reduce the geometric risk.
"""
function lower_idx(lambdas::Vector{Float64}, γ::Float64; α::Float64=1.0)
    @assert γ > 0.0
    @assert 0.0 < α <= 1.0

    # Retrieve the critical signal threshold s* where f(s*) = 1/2
    s_star = signal_thres(γ; α=α)

    # Translate s* back into the eigenvalue scale: λ = 1 + s * √γ
    λ_star = 1.0 + s_star * sqrt(γ)

    # Find the last index strictly greater than the optimal threshold
    d_star = 0
    @inbounds for l in eachindex(lambdas)
        if lambdas[l] > λ_star
            d_star = l
        else
            break # lambdas are non-increasing, so we can safely break early
        end
    end

    return d_star
end