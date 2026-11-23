module LungMethodSupport

using GeometricPCA: SemiOrthogonal, geometric_risk_sequential
using LinearAlgebra: svd
using Random: MersenneTwister, randperm
using Statistics: mean

export LUNG_DATASETS,
       read_lung_matrix,
       centered_lung_matrix,
       split_lung_bases,
       write_result_rows,
       geometricpca_ranks

const EXPERIMENTS_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(EXPERIMENTS_DIR, "datasets")

"""Descriptions of the processed lung datasets used in the comparisons."""
const LUNG_DATASETS = [
    (name = "Lung_combined", path = joinpath(DATA_DIR, "Lung_combined.csv")),
    (name = "Lung_JinWang2016_p251_n181",
     path = joinpath(DATA_DIR, "Lung_JinWang2016_p251_n181.csv")),
]

"""
    read_lung_matrix(path) -> Matrix{Float64}

Read a comma-delimited processed lung dataset into an observations-by-features
matrix. If the file has the `cancer_type,split` metadata columns, those columns
are omitted. The feature header is optional.
"""
function read_lung_matrix(path::AbstractString)
    isfile(path) || throw(ArgumentError("Lung data file not found: $path"))
    lines = filter(!isempty, strip.(readlines(path)))
    isempty(lines) && throw(ArgumentError("Lung data file is empty: $path"))

    first_fields = split(lines[1], ',')
    has_header = tryparse(Float64, strip(first(first_fields))) === nothing
    header = has_header ? strip.(first_fields) : String[]
    first_data_line = has_header ? 2 : 1
    first_data_line <= length(lines) ||
        throw(ArgumentError("Lung data file has a header but no observations: $path"))

    metadata_columns = has_header && length(header) >= 2 &&
        header[1:2] == ["cancer_type", "split"] ? 2 : 0
    first_feature = metadata_columns + 1
    n_features = length(first_fields) - metadata_columns
    n_features > 0 || throw(ArgumentError("No numeric features found in $path"))

    n_observations = length(lines) - first_data_line + 1
    X = Matrix{Float64}(undef, n_observations, n_features)
    for (i, line_index) in enumerate(first_data_line:length(lines))
        fields = split(lines[line_index], ',')
        length(fields) == length(first_fields) || throw(ArgumentError(
            "Inconsistent column count in $path at line $line_index",
        ))
        for j in 1:n_features
            value = tryparse(Float64, strip(fields[first_feature + j - 1]))
            isnothing(value) && throw(ArgumentError(
                "Non-numeric feature value in $path at line $line_index, column $(first_feature + j - 1)",
            ))
            X[i, j] = value
        end
    end
    return X
end

"Read a processed lung dataset and subtract each feature's sample mean." 
function centered_lung_matrix(path::AbstractString)
    X = read_lung_matrix(path)
    return X .- mean(X; dims = 1)
end

"""
    split_lung_bases(X; d=20, alpha=0.5, seed=523)

Randomly split the observations, compute the top `d` right singular vectors
for each fold, and return `(reference_basis, validation_basis, split_point)`.
The split follows the experiment scripts' `round(Int, n * alpha)` convention.
"""
function split_lung_bases(X::AbstractMatrix; d::Integer = 20,
                          alpha::Real = 0.5, seed::Integer = 523)
    n, p = size(X)
    d >= 1 || throw(ArgumentError("d must be positive"))
    0 < alpha < 1 || throw(ArgumentError("alpha must lie strictly between 0 and 1"))

    split_point = round(Int, n * alpha)
    split_point >= d || throw(ArgumentError(
        "Reference fold has $split_point observations, fewer than requested dimension $d",
    ))
    n - split_point >= d || throw(ArgumentError(
        "Validation fold has $(n - split_point) observations, fewer than requested dimension $d",
    ))

    indices = randperm(MersenneTwister(seed), n)
    X_reference = Matrix(view(X, indices[1:split_point], :))
    X_validate = Matrix(view(X, indices[split_point + 1:end], :))

    # Compute PCA around each fold's own feature means.
    X_reference .-= mean(X_reference; dims = 1)
    X_validate .-= mean(X_validate; dims = 1)

    # PCA bases are feature-space (right singular) vectors, matching the
    # common ambient feature dimension used by GeometricPCA's risk functions.
    reference_basis = svd(X_reference).V[:, 1:d]
    validation_basis = svd(X_validate).V[:, 1:d]
    return reference_basis, validation_basis, split_point
end

"Write a vector of named tuples as a CSV file, creating its parent directory." 
function write_result_rows(path::AbstractString, rows::AbstractVector{<:NamedTuple})
    isempty(rows) && throw(ArgumentError("Cannot write a result table with no rows"))
    columns = collect(keys(first(rows)))
    all(collect(keys(row)) == columns for row in rows) ||
        throw(ArgumentError("All result rows must have the same columns in the same order"))

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join(string.(values(row)), ','))
        end
    end
    return path
end

"""Return the chordal and Bures-Wasserstein selected ranks for a data matrix."""
function geometricpca_ranks(X::AbstractMatrix; d::Integer = 20,
                            alpha::Real = 0.5, seed::Integer = 523)
    reference, validation, _ = split_lung_bases(X; d, alpha, seed)
    U = SemiOrthogonal(reference)
    V = SemiOrthogonal(validation)
    chordal = geometric_risk_sequential(U, V; p = 2, scaled = true)
    bw = geometric_risk_sequential(U, V; p = 1, scaled = true)
    return (; chordal = argmin(chordal), bw = argmin(bw))
end

end # module LungMethodSupport
