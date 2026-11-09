# Run from the repository root with:
#   julia --project=experiments experiments/analyze_genome.jl

using GeometricPCA
using LinearAlgebra
using Random
using Statistics
using CairoMakie
using JLD2

const DATA_DIR = joinpath(@__DIR__, "datasets")
const FIGURE_DIR = joinpath(@__DIR__, "figures")
const INPUT_PATH = joinpath(DATA_DIR, "1000G_matrix_BEMA_preprocessing.jld2")
const RESULT_PATH = joinpath(DATA_DIR, "1000_genomes_geometricpca_risk.csv")
const FIGURE_PATH = joinpath(FIGURE_DIR, "1000_genomes_geometricpca_risk.png")

const DATASET_LABEL = "1000 Genomes"
const ALPHA = 0.5
const D_REF = 30
const RANDOM_SEED = 123

"Return the leading feature-space right singular vectors of a centered fold."
function leading_feature_basis(X::AbstractMatrix{<:Real}, d::Int)
    n, p = size(X)
    d <= min(n, p) || error("Requested D_REF=$d exceeds fold rank bound min(n=$n, p=$p)")
    centered = Matrix{Float64}(X .- mean(X; dims=1))
    decomposition = svd(centered; full=false)
    return SemiOrthogonal(Matrix(adjoint(decomposition.Vt[1:d, :])))
end

function main()
    isfile(INPUT_PATH) || error("Genome matrix not found: $INPUT_PATH")
    mkpath(FIGURE_DIR)

    genome_data = load(INPUT_PATH)
    haskey(genome_data, "X") || error("JLD2 file does not contain an X matrix")
    X = Matrix{Float64}(genome_data["X"])
    n, p = size(X)

    split_point = round(Int, n * ALPHA)
    n_reference, n_validate = split_point, n - split_point
    min(n_reference, n_validate) >= D_REF || error("Too few samples per fold for D_REF=$D_REF")

    indices = randperm(MersenneTwister(RANDOM_SEED), n)
    reference = view(X, indices[1:split_point], :)
    validate = view(X, indices[split_point+1:end], :)
    U_reference = leading_feature_basis(reference, D_REF)
    V_validate = leading_feature_basis(validate, D_REF)

    risk_chordal = geometric_risk_sequential(U_reference, V_validate; p=2, scaled=true)
    risk_bw = geometric_risk_sequential(U_reference, V_validate; p=1, scaled=true)

    println("$DATASET_LABEL: n=$n, p=$p, folds=$n_reference/$n_validate")
    println("  Chordal risk minimum at k=$(argmin(risk_chordal))")
    println("  BW risk minimum at k=$(argmin(risk_bw))")

    open(RESULT_PATH, "w") do io
        println(io, "dataset,n,p,metric,candidate_dimension,scaled_risk")
        for (metric, risk) in (("chordal", risk_chordal), ("BW", risk_bw))
            for (k, value) in enumerate(risk)
                println(io, "$DATASET_LABEL,$n,$p,$metric,$k,$value")
            end
        end
    end

    fig = Figure(size=(900, 500), fontsize=18)
    ax = Axis(fig[1, 1],
        title="$DATASET_LABEL: n=$n, p=$p",
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
    save(FIGURE_PATH, fig)

    println("Saved 1000 Genomes risk plot to $FIGURE_PATH")
    println("Saved 1000 Genomes risk values to $RESULT_PATH")
end

main()
