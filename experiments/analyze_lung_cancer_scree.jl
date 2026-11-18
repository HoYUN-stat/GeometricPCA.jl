# Run from the repository root with:
#   julia --project=experiments experiments/analyze_lung_cancer_scree.jl

using CairoMakie
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "src", "LungMethodSupport.jl"))
using .LungMethodSupport

const data_dir = joinpath(@__DIR__, "datasets")
const fig_dir = joinpath(@__DIR__, "figures")
const input_file = joinpath(data_dir, "Lung_JinWang2016_p251_n181.csv")
const fig_file = joinpath(fig_dir, "lung_cancer_scree_first_100.png")
const n_components = 100

function main()
    isfile(input_file) || error("Processed lung matrix not found: $input_file")
    X = read_lung_matrix(input_file)
    n, p = size(X)

    # Center each feature before computing PCA. Singular values squared are
    # proportional to the variance explained by the corresponding component.
    centered = X .- mean(X; dims=1)
    singular_values = svdvals(centered)
    n_available = min(length(singular_values), n_components)
    n_available == n_components || error(
        "Requested $n_components components, but the centered matrix has rank bound $(length(singular_values))",
    )

    explained_variance = singular_values .^ 2 ./ sum(abs2, singular_values)
    components = 1:n_components
    cumulative_variance = cumsum(explained_variance[components])

    mkpath(fig_dir)
    fig = Figure(size=(1000, 700), fontsize=18)

    ax_scree = Axis(fig[1, 1],
        title="Lung cancer PCA scree plot (n=$n, p=$p)",
        xlabel="Principal Component",
        ylabel="Variance Explained (%)",
    )
    lines!(ax_scree, components, 100 .* explained_variance[components]; linewidth=2, color=:royalblue)
    scatter!(ax_scree, components, 100 .* explained_variance[components]; markersize=5, color=:royalblue)
    ax_scree.xticks = 0:10:100

    ax_cumulative = Axis(fig[2, 1],
        xlabel="Principal Component",
        ylabel="Cumulative Variance Explained (%)",
    )
    lines!(ax_cumulative, components, 100 .* cumulative_variance; linewidth=2, color=:darkorange)
    scatter!(ax_cumulative, components, 100 .* cumulative_variance; markersize=5, color=:darkorange)
    ax_cumulative.xticks = 0:10:100
    ylims!(ax_cumulative, 0, 100)

    save(fig_file, fig)
    println("Lung cancer PCA: n=$n, p=$p")
    println("Variance explained by the first $n_components components: $(round(100 * cumulative_variance[end]; digits=2))%")
    println("Saved scree plot to $fig_file")
end

main()
