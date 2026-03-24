module PlotUtils

using CairoMakie

export plot_risk_curve!, plot_scree!

function plot_risk_curve!(ax, risk_curve::Vector{Float64}, true_d::Int; label::String="Risk", color=nothing, linestyle=nothing)
    D_max = length(risk_curve)
    estimated_d = argmin(risk_curve)

    lines!(ax, 1:D_max, risk_curve, linewidth=3, label=label, color=color, linestyle=linestyle)
    scatter!(ax, 1:D_max, risk_curve, markersize=12, color=color)
    scatter!(ax, [estimated_d], [risk_curve[estimated_d]],
        color=:red, marker=:star5, markersize=20)

    # Mark true dimension
    vlines!(ax, [true_d], color=:black, linestyle=:dash, linewidth=2)
end

function plot_scree!(ax, eigenvalues::Vector{Float64}, true_d::Int; label::String="Eigenvalues", color=nothing, linestyle=nothing)
    num_eigenvalues = length(eigenvalues)
    # ax.yscale = log10 # Use a log scale to better see the "elbow"

    # Clip non-positive eigenvalues to a small positive number to avoid log(0) error
    clipped_eigenvalues = max.(eigenvalues, eps(Float64))

    # Plot sample eigenvalues
    lines!(ax, 1:num_eigenvalues, clipped_eigenvalues, linewidth=3, color=color, label=label, linestyle=linestyle)
    scatter!(ax, 1:num_eigenvalues, clipped_eigenvalues, markersize=12, color=color)

    # Mark true dimension
    vlines!(ax, [true_d], color=:black, linestyle=:dash, linewidth=2)
end

end # module
