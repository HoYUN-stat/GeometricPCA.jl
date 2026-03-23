using LinearAlgebra

# Forward-declare SemiOrthogonal for use in function signatures
struct SemiOrthogonal{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    matrix::M
end

"""
    orthogonalize(A::AbstractMatrix) -> SemiOrthogonal

Constructs a `SemiOrthogonal` matrix from a generic matrix `A` by explicitly
orthogonalizing its columns using QR decomposition.
"""
function orthogonalize(A::AbstractMatrix)
    p, d = size(A)
    @assert p >= d "The number of rows must be greater than or equal to the number of columns (p=$p, d=$d)."

    Q_full = qr(A).Q
    semi_ortho_matrix = Q_full[:, 1:d]
    return SemiOrthogonal(semi_ortho_matrix)
end

"""
    TallSkinnyIdentity{T} <: AbstractMatrix{T}

A lazy representation of a `p x d` semi-orthogonal matrix that is equivalent
to the first `d` columns of the `p x p` identity matrix. It avoids allocating
the full `p x d` matrix in memory.
"""
struct TallSkinnyIdentity{T} <: AbstractMatrix{T}
    p::Int
    d::Int

    function TallSkinnyIdentity(p::Int, d::Int; T::Type=Float64)
        @assert p >= d "p ($p) must be greater than or equal to d ($d)."
        new{T}(p, d)
    end
end

Base.size(A::TallSkinnyIdentity) = (A.p, A.d)
Base.eltype(::TallSkinnyIdentity{T}) where {T} = T

function Base.getindex(A::TallSkinnyIdentity{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(A, i, j)
    if i == j && i <= A.d
        return one(T)
    else
        return zero(T)
    end
end

# Efficient multiplication for Adjoint(TallSkinnyIdentity) * AbstractMatrix
function LinearAlgebra.mul!(C::AbstractMatrix, At::Adjoint{<:Any,<:TallSkinnyIdentity}, B::AbstractMatrix)
    A = At.parent
    p_A, d_A = size(A)
    p_B, d_B = size(B)
    p_C, d_C = size(C)

    @assert p_A == p_B "Inner dimensions must match for multiplication"
    @assert d_A == p_C && d_B == d_C "Output dimensions must match"

    # This is equivalent to C = B[1:d_A, :]
    view(C, :, :) .= view(B, 1:d_A, :)
    return C
end

"""
    SemiOrthogonal{T, M<:AbstractMatrix{T}} <: AbstractMatrix{T}

A `p x d` semi-orthogonal matrix `V`, representing a matrix with orthonormal columns
such that `V' * V = I`.

This struct is a wrapper around any `AbstractMatrix` and **trusts** that the wrapped
matrix has orthonormal columns. To create a `SemiOrthogonal` from a matrix that
is not yet orthogonal, use the `orthogonalize()` function.

# Fields
- `matrix::M`: The underlying `p x d` matrix.

# Constructors
- `SemiOrthogonal(A::AbstractMatrix)`: Wraps an existing matrix, assuming it is already semi-orthogonal.
- `SemiOrthogonal(p::Int, d::Int)`: Lazily represents the `p x d` tall-skinny identity matrix.
"""
# The lazy constructor for SemiOrthogonal
function SemiOrthogonal(p::Int, d::Int; T::Type=Float64)
    return SemiOrthogonal(TallSkinnyIdentity(p, d; T=T))
end

# Basic matrix interface to make the struct behave like a matrix
Base.size(A::SemiOrthogonal) = size(A.matrix)
Base.getindex(A::SemiOrthogonal, I...) = getindex(A.matrix, I...)
Base.parent(A::SemiOrthogonal) = A.matrix
LinearAlgebra.transpose(A::SemiOrthogonal) = transpose(A.matrix)
LinearAlgebra.adjoint(A::SemiOrthogonal) = adjoint(A.matrix)


"""
    schatten_norm_pow(U::SemiOrthogonal, V::SemiOrthogonal; p::Int=2) -> Float64


Computes the `p`-th power of the Schatten `p`-norm of the matrix product `U' * V`.

Given two semi-orthogonal matrices, `U` (of size `p x d1`) and `V` (of size `p x d2`),
this function first computes the `d1 x d2` matrix `C = U' * V`, and then finds the
`p`-th power of its Schatten `p`-norm. This is done for numerical precision.

# Arguments
- `U::SemiOrthogonal`: The first semi-orthogonal matrix.
- `V::SemiOrthogonal`: The second semi-orthogonal matrix.

# Keyword Arguments
- `p::Int`: The order of the Schatten norm. Must be `1` or `2`. Defaults to `2`.
  - `p=1`: Returns the Nuclear norm (sum of singular values).
  - `p=2`: Returns the squared Frobenius norm (`sum(σ²)`, where `σ` are singular values).

# Returns
- `Float64`: The `p`-th power of the Schatten `p`-norm of `U' * V`.
"""
function schatten_norm_pow(U::SemiOrthogonal, V::SemiOrthogonal; p::Int=2)
    @assert p in (1, 2) "p must be 1 or 2"
    @assert size(U, 1) == size(V, 1) "Input matrices must have the same number of rows (p)."

    # Form the product C = U' * V. The result is a d1 x d2 matrix.
    C = adjoint(U) * V

    if p == 2
        # For p=2, the squared Frobenius norm is just the sum of squares of all elements.
        # This is more efficient than computing singular values.
        return sum(abs2, C)
    else # p == 1
        # Get the singular values of C for the nuclear norm.
        σ = svdvals(C)
        # Schatten 1-norm (Nuclear norm), which is sum(σ)^1.
        return sum(σ)
    end
end


"""
    geometric_risk(U::SemiOrthogonal, V::SemiOrthogonal; p::Int=2, scaled::Bool=false) -> Float64


Computes the distance between two semi-orthogonal matrices.

# Arguments
- `U::SemiOrthogonal`: The reference semi-orthogonal matrix.
- `V::SemiOrthogonal`: the validation semi-orthogonal matrix.

# Keyword Arguments
- `p::Int`: The order of the Schatten norm. Must be `1` or `2`. Defaults to `2`.
  - `p=1`: Bures-Wasserstein distance
  - `p=2`: Chordal distance
- `scaled::Bool`: If true, scaled by `size(U, 2)`. Defaults to `false`.
"""
function geometric_risk(U::SemiOrthogonal, V::SemiOrthogonal; p::Int=2, scaled::Bool=false)
    @assert p in (1, 2) "p must be 1 or 2"
    @assert size(U, 1) == size(V, 1) "Input matrices must have the same number of rows (p)."

    D = size(U, 2)
    k = size(V, 2)
    sch = schatten_norm_pow(U, V; p=p)
    r = k + D - 2.0 * sch

    if scaled
        return r / D
    else
        return r
    end
end

function geometric_risk(p_dim::Int, D::Int, V::SemiOrthogonal; kwargs...)
    U = SemiOrthogonal(p_dim, D)
    return geometric_risk(U, V; kwargs...)
end


"""
    geometric_risk_sequential(U::SemiOrthogonal, V::SemiOrthogonal; p::Int=2, scaled::Bool=false) -> Vector{Float64}

Computes the risk curve between a reference matrix `U` and sequential truncations of `V`.

The risk at index `k` of the returned vector is `geometric_risk(U, V[:, 1:k])`.

# Arguments
- `U::SemiOrthogonal`: The reference semi-orthogonal matrix.
- `V::SemiOrthogonal`: The semi-orthogonal matrix to be sequentially evaluated.

# Keyword Arguments
- `p::Int`: The order of the Schatten norm. Must be `1` or `2`. Defaults to `2`.
- `scaled::Bool`: If true, the risk at each step `k` is scaled by `size(U, 2)`. Defaults to `false`.
"""
function geometric_risk_sequential(U::SemiOrthogonal, V::SemiOrthogonal; p::Int=2, scaled::Bool=false)
    @assert p in (1, 2) "p must be 1 or 2"
    p_dim_U, D = size(U)
    p_dim_V, k_V = size(V)
    @assert p_dim_U == p_dim_V "Two matrices must have the same number of rows (p)."

    risk_curve = zeros(Float64, D)

    if p == 2
        # This implementation is optimized for p=2 (squared Frobenius norm).
        # It avoids recomputing matrix products and SVDs in the loop.
        C = adjoint(U) * V # Pre-calculate the full D x k_V product
        # Calculate the cumulative sum of squared norms down the rows of C
        sch_pows_2 = cumsum(vec(sum(abs2, C; dims=2)))

        for k in 1:D
            risk = k + k_V - 2.0 * sch_pows_2[k]
            risk_curve[k] = scaled ? risk / D : risk
        end
    else # p == 1
        # This is less efficient as it re-computes the SVD for each k.
        for k in 1:D
            U_k = SemiOrthogonal(view(parent(U), :, 1:k))
            sch = schatten_norm_pow(U_k, V; p=1)
            risk = k + k_V - 2.0 * sch
            risk_curve[k] = scaled ? risk / D : risk
        end
    end
    return risk_curve
end

function geometric_risk_sequential(p_dim::Int, D::Int, V::SemiOrthogonal; kwargs...)
    @assert p_dim == size(V, 1) "Two matrices must have the same number of rows (p)."
    U = SemiOrthogonal(p_dim, D)
    return geometric_risk_sequential(U, V; kwargs...)
end

