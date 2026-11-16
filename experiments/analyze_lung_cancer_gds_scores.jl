# Run from the repository root with:
#   julia --project=experiments experiments/analyze_lung_cancer_gds_scores.jl

using LinearAlgebra
using Random
using Statistics
using Distributions: Chisq, cdf

const data_dir = joinpath(@__DIR__, "datasets")
const input_file = joinpath(data_dir, "Lung_combined.csv")
const result_file = joinpath(data_dir, "lung_cancer_penalized_scores.csv")

const D = 89
const epsilon = 0.005
const alpha = 0.5
const seed = 523

"CDF of max_{1 <= k <= D} mean(Y[1:k]), for independent Y[j] ~ Chisq(D)."
function h_cdf(x::Real, d::Integer)
    d >= 1 || throw(ArgumentError("D must be positive"))
    x <= 0 && return 0.0
    probabilities = [cdf(Chisq(k * d), k * x) for k in 1:d]
    recurrence = zeros(d + 1)
    recurrence[1] = 1.0
    for m in 1:d
        recurrence[m + 1] = sum(probabilities[k] * recurrence[m - k + 1] for k in 1:m) / m
    end
    return clamp(recurrence[end], 0.0, 1.0)
end

function h_quantile(d::Integer, probability::Real)
    0 < probability < 1 || throw(ArgumentError("probability must lie in (0, 1)"))
    lo, hi = 0.0, max(2.0 * d, 1.0)
    while h_cdf(hi, d) < probability
        hi *= 2
    end
    for _ in 1:70
        mid = (lo + hi) / 2
        if h_cdf(mid, d) < probability
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

"Read the processed lung CSV, skipping its cancer-type and split columns."
function read_lung_matrix(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("Lung data file is empty: $path")

    header = split(lines[1], ',')
    length(header) >= 3 || error("Expected two covariate columns and at least one feature")
    X = Matrix{Float64}(undef, length(lines) - 1, length(header) - 2)

    for (sample, line) in enumerate(@view lines[2:end])
        values = split(line, ',')
        length(values) == length(header) || error("Inconsistent number of columns in lung CSV at row $sample")
        for feature in 1:size(X, 2)
            X[sample, feature] = parse(Float64, values[feature + 2])
        end
    end
    return X
end

"Return the leading feature-space right singular vectors of a centered fold."
function leading_feature_basis(X::AbstractMatrix{<:Real}, d::Int)
    n, p = size(X)
    d <= min(n, p) || error("Requested D=$d exceeds fold rank bound min(n=$n, p=$p)")
    centered = Matrix{Float64}(X .- mean(X; dims=1))
    decomposition = svd(centered; full=false)
    return Matrix(adjoint(decomposition.Vt[1:d, :]))
end

function main()
    isfile(input_file) || error("Processed lung matrix not found: $input_file")
    X = read_lung_matrix(input_file)
    n, p = size(X)
    split_point = round(Int, n * alpha)
    min(split_point, n - split_point) >= D || error("Too few samples per fold for D=$D")

    hcut = h_quantile(D, 1 - epsilon)
    t = hcut / p

    indices = randperm(MersenneTwister(seed), n)
    reference = view(X, indices[1:split_point], :)
    validate = view(X, indices[split_point+1:end], :)
    U_reference = leading_feature_basis(reference, D)
    V_validate = leading_feature_basis(validate, D)
    C = U_reference' * V_validate

    q = vec(sum(abs2, C; dims=1))
    nuclear_prefix = [sum(svdvals(C[:, 1:k])) for k in 1:D]
    theta_scores = vcat(0.0, 2 .* collect(1:D) .* t .- 2 .* cumsum(q))
    wasserstein_scores = vcat(0.0, 2 .* collect(1:D) .* sqrt(t) .- 2 .* nuclear_prefix)
    selected_theta = argmin(theta_scores) - 1
    selected_wasserstein = argmin(wasserstein_scores) - 1

    open(result_file, "w") do io
        println(io, "candidate_dimension,theta_score,wasserstein_score,selected_theta_dimension,selected_wasserstein_dimension")
        for k in 0:D
            println(io, "$k,$(theta_scores[k + 1]),$(wasserstein_scores[k + 1]),$selected_theta,$selected_wasserstein")
        end
    end

    println("Lung cancer GDS scores: n=$n, p=$p, folds=$split_point/$(n - split_point)")
    println("  D=$D, epsilon=$epsilon, H_D quantile=$hcut, t=H_D/p=$t")
    println("  Seed: $seed")
    println("  Theta selected dimension: $selected_theta")
    println("  Wasserstein selected dimension: $selected_wasserstein")
    println("Saved score curves to $result_file")
end

main()
