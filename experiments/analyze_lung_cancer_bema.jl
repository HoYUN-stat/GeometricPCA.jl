# Run from the repository root with:
#   julia --project=experiments experiments/analyze_lung_cancer_bema.jl
#
# Julia implementation of SQM_Gamma from experiments/src/BEMA.R (Ke et al.,
# 2023). The input is used without centering, matching the R implementation.

using LinearAlgebra
using Random
using Statistics
using Distributions: Gamma

const data_file = joinpath(@__DIR__, "datasets", "Lung_JinWang2016_p251_n181.csv")
const result_file = joinpath(@__DIR__, "datasets", "lung_cancer_bema_jinwang.csv")
const alpha = 0.1
const seed = 523
const optimization_replicates = 10
const final_replicates = 500

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

"Eigenvalues of X'X/n in decreasing order, using the smaller Gram matrix."
function sample_eigenvalues(X::AbstractMatrix{<:Real})
    n, p = size(X)
    if p <= n
        return reverse(eigvals(Symmetric(Matrix(X' * X / n))))
    end
    return reverse(eigvals(Symmetric(Matrix(X * X' / n))))
end

"Draw the p-dimensional diagonal-Gamma Gaussian null model used by BEMA."
function null_eigenvalues(rng::AbstractRNG, n::Int, p::Int, shape::Float64)
    variances = rand(rng, Gamma(shape, 1.0), p)
    X = randn(rng, n, p)
    X .*= reshape(sqrt.(variances), 1, p)
    values = sample_eigenvalues(X)
    if p > n
        return vcat(values, zeros(p - n))
    end
    return values
end

function gamma_loss(rng, observed, n, p, shape, indices)
    simulated = zeros(optimization_replicates, p)
    for rep in axes(simulated, 1)
        simulated[rep, :] .= null_eigenvalues(rng, n, p, shape)
    end
    mean_spectrum = vec(mean(simulated; dims=1))
    scale = dot(observed[indices], mean_spectrum[indices]) / sum(abs2, mean_spectrum[indices])
    return sum(abs2, scale .* mean_spectrum[indices] .- observed[indices])
end

"Golden-section bounded minimization corresponding to R's optimize()."
function bounded_minimum(f, lower, upper; tolerance=sqrt(eps(Float64)))
    golden = (3 - sqrt(5.0)) / 2
    x1 = lower + golden * (upper - lower)
    x2 = upper - golden * (upper - lower)
    f1, f2 = f(x1), f(x2)
    while upper - lower > tolerance * (abs(x1) + abs(x2))
        if f1 <= f2
            upper, x2, f2 = x2, x1, f1
            x1 = lower + golden * (upper - lower)
            f1 = f(x1)
        else
            lower, x1, f1 = x1, x2, f2
            x2 = upper - golden * (upper - lower)
            f2 = f(x2)
        end
    end
    return f1 <= f2 ? x1 : x2
end

function main()
    isfile(data_file) || error("Processed lung matrix not found: $data_file")
    X = read_lung_matrix(data_file)
    n, p = size(X)
    m = min(n, p)
    indices = floor(Int, m * alpha):floor(Int, m * (1 - alpha))
    observed = vcat(sample_eigenvalues(X), zeros(max(0, p - n)))

    # BEMA's R code redraws at each objective evaluation. A fixed seed makes
    # this stochastic optimization and its final reported estimate reproducible.
    rng = MersenneTwister(seed)
    shape = bounded_minimum(0.1, 50.0) do proposal
        gamma_loss(rng, observed, n, p, proposal, indices)
    end

    simulated = Matrix{Float64}(undef, final_replicates, p)
    for rep in 1:final_replicates
        simulated[rep, :] .= null_eigenvalues(rng, n, p, shape)
    end
    mean_spectrum = vec(mean(simulated; dims=1))
    upper_spectrum = [quantile(view(simulated, :, j), 0.9) for j in 1:p]
    scale = dot(observed[indices], mean_spectrum[indices]) / sum(abs2, mean_spectrum[indices])
    threshold = maximum(scale .* upper_spectrum)
    estimated_k = count(>(threshold), observed)

    println("JinWang lung BEMA SQM_Gamma: n=$n, p=$p")
    println("  alpha=$alpha, fitted Gamma shape=$shape")
    println("  90th-percentile null spectral edge=$threshold")
    println("  Estimated K=$estimated_k")

    open(result_file, "w") do io
        println(io, "n,p,alpha,gamma_shape,scale,null_edge,estimated_K")
        println(io, "$n,$p,$alpha,$shape,$scale,$threshold,$estimated_k")
    end
    println("Saved estimate to $result_file")
end

main()
