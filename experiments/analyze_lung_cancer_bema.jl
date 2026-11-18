# Run from the repository root with:
#   julia --project=experiments experiments/analyze_lung_cancer_bema.jl
#
# Julia implementation of SQM_Gamma from experiments/src/BEMA.R (Ke et al.,
# 2023). The input is used without centering, matching the R implementation.

using LinearAlgebra
using Random
using Statistics
using Distributions: Gamma

if !isdefined(@__MODULE__, :LungMethodSupport)
    include(joinpath(@__DIR__, "src", "LungMethodSupport.jl"))
end
using .LungMethodSupport

const BEMA_RESULT_FILE = joinpath(@__DIR__, "datasets", "lung_cancer_bema_results.csv")
const alpha = 0.1
const seed = 523
const optimization_replicates = 10
const final_replicates = 500

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

function bema_rank(X; seed = 523)
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

    return (; n, p, estimated_K = estimated_k, gamma_shape = shape,
            scale, null_edge = threshold)
end

function run_bema_analysis()
    rows = NamedTuple[]
    for dataset in LUNG_DATASETS
        isfile(dataset.path) || error("Processed lung matrix not found: $(dataset.path)")
        X = read_lung_matrix(dataset.path)
        estimate = bema_rank(X; seed)
        push!(rows, (; dataset = dataset.name, n = estimate.n, p = estimate.p,
                     method = "BEMA", selected_dimension = estimate.estimated_K,
                     gamma_shape = estimate.gamma_shape, scale = estimate.scale,
                     null_edge = estimate.null_edge))
        println("$(dataset.name): BEMA SQM_Gamma dimension=$(estimate.estimated_K), n=$(estimate.n), p=$(estimate.p)")
    end

    open(BEMA_RESULT_FILE, "w") do io
        println(io, "dataset,n,p,method,selected_dimension,gamma_shape,scale,null_edge")
        for row in rows
            println(io, join(string.(values(row)), ','))
        end
    end
    println("Saved BEMA results to $BEMA_RESULT_FILE")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_bema_analysis()
end
