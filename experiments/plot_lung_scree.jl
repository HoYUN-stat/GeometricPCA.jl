# Run from the repository root with:
#   julia --project=experiments experiments/plot_lung_scree.jl

using LinearAlgebra
using Statistics
using CairoMakie

const data_dir = joinpath(@__DIR__, "datasets")
const fig_dir = joinpath(@__DIR__, "figures")
const combined_file = joinpath(data_dir, "Lung_combined.csv")
const jinwang_file = joinpath(data_dir, "Lung_JinWang2016_p251_n181.csv")
const plot_file = joinpath(fig_dir, "lung_screeplots.png")

function read_feature_matrix(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("Lung data file is empty: $path")

    header = split(lines[1], ',')
    length(header) >= 3 || error("Expected two metadata columns and at least one feature in $path")
    x = Matrix{Float64}(undef, length(lines) - 1, length(header) - 2)

    for (i, line) in enumerate(@view lines[2:end])
        values = split(line, ',')
        length(values) == length(header) || error("Inconsistent column count in $path at row $i")
        for j in axes(x, 2)
            x[i, j] = parse(Float64, values[j + 2])
        end
    end
    return x
end

function scree_values(x::AbstractMatrix{<:Real})
    n = size(x, 1)
    centered = x .- mean(x; dims=1)
    return svd(centered; full=false).S .^ 2 ./ (n - 1)
end

datasets = (
    ("Lung combined", combined_file),
    ("JinWang (251 features)", jinwang_file),
)

mkpath(fig_dir)
fig = Figure(size=(1100, 480), fontsize=16)

for (column, (label, path)) in enumerate(datasets)
    x = read_feature_matrix(path)
    eigenvalues = scree_values(x)
    n, p = size(x)

    ax = Axis(fig[1, column],
        title="$label (n=$n, p=$p)",
        xlabel="Principal component",
        ylabel="Covariance eigenvalue",
    )
    components = eachindex(eigenvalues)
    lines!(ax, components, eigenvalues; linewidth=2, color=:navy)
    scatter!(ax, components, eigenvalues; markersize=5, color=:navy)
end

save(plot_file, fig)
println("Saved lung scree plots to $plot_file")
