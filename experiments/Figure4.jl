# Run from the repository root with:
#   julia --project=experiments experiments/analyze_lung_cancer.jl

using GeometricPCA
using CairoMakie

include(joinpath(@__DIR__, "src", "LungMethodSupport.jl"))
using .LungMethodSupport

const data_dir = joinpath(@__DIR__, "datasets")
const fig_dir = joinpath(@__DIR__, "figures")
const input_file = joinpath(data_dir, "Lung_JinWang2016_p251_n181.csv")
const result_file = joinpath(data_dir, "lung_cancer_pruned_risk.csv")
const fig_file = joinpath(fig_dir, "lung_cancer_pruned_risk.png")

const dataset = "Lung cancer (JinWang2016, 251 features)"
const alpha = 0.5
const d_ref = 20
const seed = 523

function main()
    isfile(input_file) || error("Processed lung matrix not found: $input_file")
    mkpath(fig_dir)
    X = read_lung_matrix(input_file)

    n, p = size(X)
    U_reference, V_validate, split_point = split_lung_bases(X; d=d_ref, alpha, seed)
    n_reference, n_validate = split_point, n - split_point
    U_reference = SemiOrthogonal(U_reference)
    V_validate = SemiOrthogonal(V_validate)

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
