# Exact dense Gram-matrix eigendecompositions. No approximate/truncated iteration.
mutable struct PartialSpectralFrame{F}
    S::Vector{Float64}
    factor::F
    basis::Matrix{Float64}
    n::Int
    p::Int
end

function spectral_frame(X::Matrix{Float64}; leading::Union{Nothing,Int} = nothing, lazy::Bool = false, partial::Bool = true)
    n, p = size(X)
    r = min(n, p)
    gram = n >= p ? BLAS.syrk('U', 'T', 1.0, X) : BLAS.syrk('U', 'N', 1.0, X)
    if lazy && leading === nothing && partial
        # hetrd! checks the entire backing array, including the triangle that
        # syrk intentionally leaves uninitialized.
        @inbounds for j in 1:r, i in (j+1):r
            gram[i, j] = gram[j, i]
        end
        # Retain all eigenvalues, but back-transform only the eigenvectors used
        # by DDPA or a selected subspace. This does not cap any method's rank.
        factor = hessenberg!(Symmetric(gram, :U))
        singular_values = sqrt.(max.(reverse(eigvals(factor.H)), 0.0))
        return PartialSpectralFrame(singular_values, factor, zeros(r, 0), n, p)
    end
    index = leading === nothing ? (1:r) : (r-leading+1:r)
    eig = eigen!(Symmetric(gram, :U), index)
    singular_values = sqrt.(max.(reverse(eig.values), 0.0))
    basis = reverse(eig.vectors; dims = 2)
    V = n >= p ? basis : lazy ? nothing : (X' * basis) ./ reshape(singular_values, 1, :)
    return (; S = singular_values, V, basis)
end

function right_vectors(frame::PartialSpectralFrame, X, indices)
    isempty(indices) && return zeros(frame.p, 0)
    last_index = maximum(indices)
    r = length(frame.S)
    if last_index > size(frame.basis, 2)
        target = min(r, max(32, last_index, 2 * size(frame.basis, 2)))
        small = eigen(frame.factor.H, (r-target+1):r)
        frame.basis = frame.factor.Q * reverse(small.vectors; dims = 2)
    end
    selected = frame.basis[:, indices]
    return frame.n >= frame.p ? selected : (X' * selected) ./ reshape(frame.S[indices], 1, :)
end

function right_vectors(frame, X, indices)
    return frame.V === nothing ? (X' * frame.basis[:, indices]) ./ reshape(frame.S[indices], 1, :) : frame.V[:, indices]
end

function ddpa_from_frame(eigenvalues, frame, X, column_variances, gamma)
    residual_variances = copy(column_variances)
    selected = 0
    for j in eachindex(eigenvalues)
        edge = upper_edge(residual_variances, gamma)
        if eigenvalues[j] > edge
            selected += 1
            v = right_vectors(frame, X, j:j)
            residual_variances .-= eigenvalues[j] .* vec(abs2.(v))
            residual_variances .= max.(residual_variances, 0.0)
        else
            break
        end
    end
    return selected
end
