using Random, Statistics

"""
    generate_spiked_data!(
        X::AbstractMatrix{Float64},
        lambdas::Vector{Float64};
        noise_var::Float64=1.0,
        rng::AbstractRNG=Random.default_rng(),
    )

Generate spiked data in-place, filling a pre-allocated matrix `X`.

The true eigenvectors are the standard orthonormal basis vectors. The variance of the
`j`-th feature is `lambdas[j]` if `j <= d` and `noise_var` otherwise,
where `d` is the length of `lambdas`.

# Arguments
- `X::AbstractMatrix{Float64}`: The `n x p` matrix to fill with generated data.
- `lambdas::Vector{Float64}`: A vector of length `d` containing the variances of the spiked dimensions.

# Keyword Arguments
- `noise_var::Float64`: The variance of the noise dimensions. Defaults to `1.0`.
- `rng::AbstractRNG`: Random number generator. Defaults to `Random.default_rng()`.
"""
function generate_spiked_data!(
    X::AbstractMatrix{Float64},
    lambdas::Vector{Float64};
    noise_var::Float64=1.0,
    rng::AbstractRNG=Random.default_rng(),
)
    n, p = size(X)
    d = length(lambdas)

    @assert d <= p "Signal length d ($d) cannot exceed the data dimension p ($p)."
    @assert minimum(lambdas) > noise_var "Spiked variances in `lambdas` must be greater than `noise_var`."

    Threads.@threads for j in 1:p
        if j <= d
            # For spiked dimensions, variance = lambdas[j]
            std_dev = sqrt(lambdas[j])
            @inbounds for i in 1:n
                X[i, j] = randn(rng) * std_dev
            end
        else
            # For non-spiked (noise) dimensions, variance = noise_var
            std_dev = sqrt(noise_var)
            @inbounds for i in 1:n
                X[i, j] = randn(rng) * std_dev
            end
        end
    end

    return X
end

"""
    generate_spiked_data(
        n::Int,
        p::Int,
        lambdas::Vector{Float64};
        noise_var::Float64=1.0,
        rng::AbstractRNG=Random.default_rng(),
    ) -> Matrix{Float64}

Generate `n` samples of `p`-dimensional data from a spiked covariance model.

The true eigenvectors are simplified to be the standard orthonormal basis (ONB) vectors.
Therefore, the data matrix `X` of size `(n, p)` is generated such that the
variance of the `j`-th feature is:
```math
Var(X_{ij}) =
    if j <= d then λ_j
    else σ^2
```
where `λ_j` is the variance from `lambdas[j]`, `d` is `length(lambdas)`,
and `σ^2` is the `noise_var`.

# Arguments
- `n::Int`: Number of samples.
- `p::Int`: Dimensionality of the data.
- `lambdas::Vector{Float64}`: A vector of length `d` containing the variances of the spiked dimensions.

# Keyword Arguments
- `noise_var::Float64`: The noise variance. Defaults to `1.0`.
- `rng::AbstractRNG`: Random number generator. Defaults to `Random.default_rng()`.

# Returns
- `Matrix{Float64}`: The `n x p` generated data matrix.

# Examples
```jldoctest
julia> using Random, Statistics;

julia> rng = MersenneTwister(42);

julia> X = generate_spiked_data(100, 4, [10.0, 5.0]; noise_var=0.5, rng=rng);

julia> round.(var(X, dims=1), digits=1)
1×4 Matrix{Float64}:
 10.2  4.8  0.5  0.5
```
"""
function generate_spiked_data(
    n::Int,
    p::Int,
    lambdas::Vector{Float64};
    noise_var::Float64=1.0,
    rng::AbstractRNG=Random.default_rng(),
)
    X = Matrix{Float64}(undef, n, p)
    generate_spiked_data!(X, lambdas; noise_var=noise_var, rng=rng)
    return X
end

"""
    generate_spiked_data(
        n::Int,
        p::Int,
        d::Int,
        lambda_value::Float64;
        noise_var::Float64=1.0,
        rng::AbstractRNG=Random.default_rng(),
    ) -> Matrix{Float64}

Generate spiked data where the first `d` dimensions have the same variance.

This is a convenience function that creates a `lambdas` vector of `[lambda_value, lambda_value, ...]`
of length `d` and calls the primary `generate_spiked_data` method.

# Arguments
- `n::Int`: Number of samples.
- `p::Int`: Dimensionality of the data.
- `d::Int`: The number of spiked dimensions.
- `lambda_value::Float64`: The variance for each of the `d` spiked dimensions.

# Keyword Arguments
- `noise_var::Float64`: The noise variance. Defaults to `1.0`.
- `rng::AbstractRNG`: Random number generator. Defaults to `Random.default_rng()`.

# Returns
- `Matrix{Float64}`: The `n x p` generated data matrix.

# Examples
```jldoctest
julia> using Random, Statistics;

julia> Random.seed!(123);

julia> Y = generate_spiked_data(100, 5, 2, 100.0; noise_var=1.0);
```
"""
function generate_spiked_data(
    n::Int,
    p::Int,
    d::Int,
    lambda_value::Float64;
    noise_var::Float64=1.0,
    rng::AbstractRNG=Random.default_rng(),
)
    lambdas = fill(lambda_value, d)
    return generate_spiked_data(n, p, lambdas; noise_var=noise_var, rng=rng)
end