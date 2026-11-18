# Run from the repository root with:
#   julia --project=experiments experiments/analyze_lung_cancer_gd.jl

using LinearAlgebra
using Statistics

if !isdefined(@__MODULE__, :LungMethodSupport)
    include(joinpath(@__DIR__, "src", "LungMethodSupport.jl"))
end
using .LungMethodSupport

const GD_RESULT_FILE = joinpath(@__DIR__, "datasets", "lung_cancer_gd_results.csv")

# Gavish and Donoho (2014), unknown-noise optimal singular-value hard threshold.
"Median of the unit-scale Marchenko-Pastur law for aspect ratio beta <= 1."
function mp_median(beta::Real)
    0 < beta <= 1 || throw(ArgumentError("beta must lie in (0, 1]"))
    sqrt_beta = sqrt(beta)

    function angular_cdf(theta)
        beta == 1 && return (theta + sin(theta)) / pi
        # Simpson integration of the MP density after x = 1 + beta -
        # 2sqrt(beta)cos(theta), which is stable for small beta.
        n_steps = 512
        h = theta / n_steps
        density(t) = 2sin(t)^2 / (pi * (1 + beta - 2sqrt_beta * cos(t)))
        total = density(0.0) + density(theta)
        for i in 1:n_steps-1
            total += (isodd(i) ? 4 : 2) * density(i * h)
        end
        return total * h / 3
    end

    lo, hi = 0.0, pi
    for _ in 1:60
        mid = (lo + hi) / 2
        if angular_cdf(mid) < 0.5
            lo = mid
        else
            hi = mid
        end
    end
    theta = (lo + hi) / 2
    return 1 + beta - 2sqrt_beta * cos(theta)
end

"Gavish-Donoho unknown-noise hard threshold rank for a centered data matrix."
function gavish_donoho_rank(X::AbstractMatrix)
    n, p = size(X)
    singular_values = svdvals(X)
    beta = min(n, p) / max(n, p)
    lambda_squared = 2 * (1 + beta) + 8 * beta / (1 + beta + sqrt(beta^2 + 14beta + 1))
    threshold_multiplier = sqrt(lambda_squared / mp_median(beta))
    threshold = threshold_multiplier * median(singular_values)
    return (; rank = count(>(threshold), singular_values), threshold,
            threshold_multiplier, beta, median_singular_value = median(singular_values))
end

function run_gd_analysis()
    rows = NamedTuple[]
    for dataset in LUNG_DATASETS
        isfile(dataset.path) || error("Lung data file not found: $(dataset.path)")
        X = centered_lung_matrix(dataset.path)
        estimate = gavish_donoho_rank(X)
        n, p = size(X)
        push!(rows, (; dataset = dataset.name, n, p, method = "Gavish-Donoho",
                     selected_dimension = estimate.rank,
                     threshold = estimate.threshold,
                     threshold_multiplier = estimate.threshold_multiplier,
                     aspect_ratio = estimate.beta))
        println("$(dataset.name): n=$n, p=$p, GD dimension=$(estimate.rank)")
    end
    write_result_rows(GD_RESULT_FILE, rows)
    println("Saved GD results to $GD_RESULT_FILE")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_gd_analysis()
end
