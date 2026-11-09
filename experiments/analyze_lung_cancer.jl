# Run from the repository root with:
#   julia --project=experiments experiments/analyze_lung_cancer.jl

using GeometricPCA
using LinearAlgebra
using Random
using Statistics
using CairoMakie

const data_dir = joinpath(@__DIR__, "datasets")
const fig_dir = joinpath(@__DIR__, "figures")
const input_file = joinpath(data_dir, "Lung_JinWang2016_p251_n181.csv")
const result_file = joinpath(data_dir, "lung_cancer_geometricpca_risk.csv")
const fig_file = joinpath(fig_dir, "lung_cancer_geometricpca_risk.png")

const dataset = "Lung cancer"
const alpha = 0.5
const d_ref = 20
const seed = 523

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
    d <= min(n, p) || error("Requested d_ref=$d exceeds fold rank bound min(n=$n, p=$p)")
    centered = Matrix{Float64}(X .- mean(X; dims=1))
    decomposition = svd(centered; full=false)
    return SemiOrthogonal(Matrix(adjoint(decomposition.Vt[1:d, :])))
end

function main()
    isfile(input_file) || error("Processed lung matrix not found: $input_file")
    mkpath(fig_dir)
    X = read_lung_matrix(input_file)
    size(X) == (181, 251) || error("Expected lung data size (181, 251), got $(size(X))")

    n, p = size(X)
    split_point = round(Int, n * alpha)
    n_reference, n_validate = split_point, n - split_point
    min(n_reference, n_validate) >= d_ref || error("Too few samples per fold for d_ref=$d_ref")

    indices = randperm(MersenneTwister(seed), n)
    reference = view(X, indices[1:split_point], :)
    validate = view(X, indices[split_point+1:end], :)
    U_reference = leading_feature_basis(reference, d_ref)
    V_validate = leading_feature_basis(validate, d_ref)

    risk_chordal = geometric_risk_sequential(U_reference, V_validate; p=2, scaled=true)
    risk_bw = geometric_risk_sequential(U_reference, V_validate; p=1, scaled=true)

    println("$dataset: n=$n, p=$p, folds=$n_reference/$n_validate")
    println("  Chordal risk minimum at k=$(argmin(risk_chordal))")
    println("  BW risk minimum at k=$(argmin(risk_bw))")

    open(result_file, "w") do io
        println(io, "dataset,n,p,metric,candidate_dimension,scaled_risk")
        for (metric, risk) in (("chordal", risk_chordal), ("BW", risk_bw))
            for (k, value) in enumerate(risk)
                println(io, "$dataset,$n,$p,$metric,$k,$value")
            end
        end
    end

    fig = Figure(size=(900, 500), fontsize=18)
    ax = Axis(fig[1, 1],
        title="$dataset: n=$n, p=$p",
        xlabel="Candidate Dimension (k)",
        ylabel="Scaled Geometric Risk",
    )
    for (risk, label, color) in (
        (risk_chordal, "Chordal (k̂=$(argmin(risk_chordal)))", :blue),
        (risk_bw, "BW (k̂=$(argmin(risk_bw)))", :orange),
    )
        candidates = eachindex(risk)
        lines!(ax, candidates, risk; linewidth=3, label=label, color=color)
        scatter!(ax, candidates, risk; markersize=8, color=color)
        best_k = argmin(risk)
        scatter!(ax, [best_k], [risk[best_k]]; color=:red, marker=:star5, markersize=18)
    end
    axislegend(ax, position=:lt)
    save(fig_file, fig)

    println("Saved lung cancer risk plot to $fig_file")
    println("Saved lung cancer risk values to $result_file")
end

main()
