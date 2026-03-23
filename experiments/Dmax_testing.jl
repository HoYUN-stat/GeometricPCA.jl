using Pkg
Pkg.activate(".")
using GeometricPCA
using CairoMakie, Random, Statistics, Dates, LinearAlgebra

# 1. Configure Simulation
println("--- Step 1: Configuring Simulation ---")
train_split_ratio = 0.5 # Ratio of data to use for the training/reference fold
signal_strength = 50.0
noise_var = 1.0
n = 200
true_d = 10
#p=50
p = 100
D_vec = collect(10:10:50)
#scaling_vec = collect(0.1:0.1:1)

risk_matrix = zeros(maximum(D_vec), p)

function plot_risk_curve!(ax, risk_curve::Vector{Float64}, true_d::Int, D::Int)
    D_max = length(risk_curve)
    estimated_d = argmin(risk_curve)

    lines!(ax, 1:D_max, risk_curve, linewidth=3, label="D =$D ")
    scatter!(ax, 1:D_max, risk_curve, markersize=12)
    scatter!(ax, [estimated_d], [risk_curve[estimated_d]],
        color=:red, marker=:star5, markersize=20)
end

# Loop over each p and D
for i in 1:length(D_vec)
    D_max = D_vec[i]
    println("Parameters: n=$n, p=$p, true_d=$true_d, D_max=$D_max, train_split_ratio=$train_split_ratio")
    risk_matrix[i, 1:D_max] = risk_estimate(n, p, true_d, D_max, train_split_ratio, signal_strength, noise_var)
end
    # Plot
    println("\n--- Step 9: Plotting Trends ---")
    fig = Figure(size=(800, 600), fontsize=25)
    fig[0, 1] = Label(fig, "PCA plot (n=$n, p=$p, D=varying, d_true=$true_d, SNR=$signal_strength)", fontsize=24, tellwidth=false)

    ax_risk = Axis(fig[1, 1],
                   title="Geometric Risk Method (Split Ratio: $train_split_ratio)",
                   xlabel="Candidate Dimension (k)",
                   ylabel="Squared Chordal Risk"
                   )
vlines!(ax_risk, [true_d], color=:black, linestyle=:dash, linewidth=2)
for i in 1:length(D_vec)
    D_max = D_vec[i]
    plot_risk_curve!(ax_risk, risk_matrix[i,1:D_max], true_d, D_max)
end

axislegend(ax_risk, position=:rb, framevisible=true)
display(fig)
# 10. Save the Figure
#println("\n--- Step 10: Saving Figure ---")
#if !isdir("figures")
    #println("Creating 'figures' directory.")
    #mkdir("figures")
#end
filename = "figures/D_effect.png"
save(filename, fig)