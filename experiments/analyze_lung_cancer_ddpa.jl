# Run from the repository root with:
#   julia --project=experiments experiments/analyze_lung_cancer_ddpa.jl

using LinearAlgebra
using Statistics

if !isdefined(@__MODULE__, :LungMethodSupport)
    include(joinpath(@__DIR__, "LungMethodSupport.jl"))
end
using .LungMethodSupport

const DDPA_RESULT_FILE = joinpath(@__DIR__, "datasets", "lung_cancer_ddpa_results.csv")

# Dobriban and Owen (2019), DDPA Algorithm 2 with its default inflation epsilon=0.
"Upper edge of the variance-profile Marčenko-Pastur null spectrum."
function ddpa_upper_edge(column_variances, gamma::Real)
    gamma > 0 || throw(ArgumentError("gamma must be positive"))
    maximum_variance = maximum(column_variances)
    maximum_variance <= 0 && return 0.0
    normalized = column_variances ./ maximum_variance
    lo, hi = 0.0, 1.0

    # Solve the rightmost-branch edge equation for the empirical column
    # variance profile, as in deterministic parallel analysis.
    for _ in 1:70
        u = (lo + hi) / 2
        derivative_term = gamma * mean((u .* normalized ./ (1 .- u .* normalized)) .^ 2)
        if derivative_term < 1
            lo = u
        else
            hi = u
        end
    end
    u = (lo + hi) / 2
    return maximum_variance * (1 / u + gamma * mean(normalized ./ (1 .- u .* normalized)))
end

"Plain deflated deterministic parallel analysis rank (DDPA, not DDPA+)."
function ddpa_rank(X::AbstractMatrix)
    n, p = size(X)
    decomposition = svd(X; full = false)
    eigenvalues = max.(decomposition.S .^ 2 ./ n, 0.0)
    V = decomposition.V
    residual_variances = vec(sum(abs2, X; dims = 1)) ./ n
    gamma = p / n
    selected, threshold = 0, NaN

    for j in eachindex(eigenvalues)
        threshold = ddpa_upper_edge(residual_variances, gamma)
        if eigenvalues[j] > threshold
            selected += 1
            residual_variances .-= eigenvalues[j] .* abs2.(view(V, :, j))
            residual_variances .= max.(residual_variances, 0.0)
        else
            break
        end
    end
    return (; rank = selected, threshold, gamma)
end

function run_ddpa_analysis()
    rows = NamedTuple[]
    for dataset in LUNG_DATASETS
        isfile(dataset.path) || error("Lung data file not found: $(dataset.path)")
        X = centered_lung_matrix(dataset.path)
        estimate = ddpa_rank(X)
        n, p = size(X)
        push!(rows, (; dataset = dataset.name, n, p, method = "DDPA",
                     selected_dimension = estimate.rank,
                     threshold = estimate.threshold,
                     threshold_multiplier = NaN,
                     aspect_ratio = estimate.gamma))
        println("$(dataset.name): n=$n, p=$p, DDPA dimension=$(estimate.rank)")
    end
    write_result_rows(DDPA_RESULT_FILE, rows)
    println("Saved DDPA results to $DDPA_RESULT_FILE")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ddpa_analysis()
end
