module PlotUtils

using CairoMakie

export plot_risk_curve!, plot_scree!

function plot_risk_curve!(ax, risk_curve::Vector{Float64}, true_d::Int, p::Int)
    D_max = length(risk_curve)
    estimated_d = argmin(risk_curve)

    lines!(ax, 1:D_max, risk_curve, linewidth=3, label="p =$p ")
    scatter!(ax, 1:D_max, risk_curve, markersize=12)
    scatter!(ax, [estimated_d], [risk_curve[estimated_d]],
        color=:red, marker=:star5, markersize=20)
end

function plot_scree!(ax, eigenvalues::Vector{Float64}, true_eigenvalues::Vector{Float64}, true_d::Int)
    num_eigenvalues = length(eigenvalues)
    # ax.yscale = log10 # Use a log scale to better see the "elbow"

    # Clip non-positive eigenvalues to a small positive number to avoid log(0) error
    clipped_eigenvalues = max.(eigenvalues, eps(Float64))
    clipped_true_eigenvalues = max.(true_eigenvalues, eps(Float64))

    # Plot sample eigenvalues
    lines!(ax, 1:num_eigenvalues, clipped_eigenvalues, linewidth=3, color=:darkorange, label="Sample")
    scatter!(ax, 1:num_eigenvalues, clipped_eigenvalues, markersize=12, color=:darkorange)

    # Plot true eigenvalues
    lines!(ax, 1:num_eigenvalues, clipped_true_eigenvalues, linewidth=3, color=:blue, label="True")
    scatter!(ax, 1:num_eigenvalues, clipped_true_eigenvalues, markersize=12, color=:blue)

    # Mark true dimension
    vlines!(ax, [true_d], color=:black, linestyle=:dash, linewidth=2)

    axislegend(ax, position=:rt)
end

end # module
