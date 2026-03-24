# cd("experiments")
# ] activate .
# dev ..
using GeometricPCA
using CairoMakie, Random, Statistics, Dates, LinearAlgebra

include("../utils/plot_utils.jl")
using .PlotUtils

# --- Main Script Execution ---

# --------------------------------------------------------------------
#          Global Parameters
# --------------------------------------------------------------------
n = 1000 # Increased slightly to ensure enough SVD components for extreme α splits
γ = 5.0
d_true = 15
D_ref = 20

p = round(Int, γ * n)

# Linear Decay
# To our favor, lambdas[d_true] ≥ thres_lambda -> Hmm...Scree plot also does well
# lambdas = collect(d_true:-1:1) .* s .* sqrt(γ) .+ thres_lambda

# Polynomial Decay
s = 4.0         # Slope
poly_deg = 1.0    # Degree of Polynomial
lambdas = collect(d_true:-1:1) .^ (poly_deg) .* s .* sqrt(γ) .+ 1.0;

# --------------------------------------------------------------------
#          Data Generation
# --------------------------------------------------------------------
Random.seed!(123);
X = generate_spiked_data(n, p, lambdas);

alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
lambda_thres = [sqrt(γ) * signal_thres(γ; α) + 1.0 for α in alphas]

true_eigenvalues = [lambdas; fill(1.0, p - d_true)][1:D_ref]

# --------------------------------------------------------------------
#          Visualization
# --------------------------------------------------------------------
fig = Figure(size=(1000, 1000), fontsize=25)
Label(fig[0, 1], "Varying Data Split Ratio α (n=$n, γ=$γ, D=$D_ref, d*=$d_true)", font=:bold, tellwidth=false)

# ax_scree = Axis(fig[1, 1], title="True Eigenvalues", xlabel="Dimension (k)", ylabel="Eigenvalue (log scale)", yscale=log10)
ax_scree = Axis(fig[1, 1], title="True Eigenvalues", xlabel="Dimension (k)", ylabel="Eigenvalue")

plot_scree!(ax_scree, true_eigenvalues, d_true; color=:black)

ax_risk = Axis(fig[2, 1], title="Sample Risk Curves", xlabel="Candidate Dimension (k)", ylabel="Scaled Chordal Risk")

colors = [:magenta, :blue, :green, :purple, :orange]

for (i, α) in enumerate(alphas)
    c = colors[i]

    # 1. Plot Threshold
    hlines!(ax_scree, [lambda_thres[i]], color=c, linestyle=:dash, linewidth=2, label="λ(γ, α = $α)")

    # 2. Split Data
    Random.seed!(123)
    idx = randperm(n)
    split_point = round(Int, n * α)
    X_train = view(X, idx[1:split_point], :)
    X_val = view(X, idx[split_point+1:end], :)

    # 3. Compute SVDs
    _, _, U_reference = svd(X_train)
    U_ref = SemiOrthogonal(U_reference[:, 1:D_ref])

    _, _, V_validate = svd(X_val)
    V_val = SemiOrthogonal(V_validate[:, 1:D_ref])

    # 4. Evaluate Geometric Risk
    scaled_risk_curve = geometric_risk_sequential(U_ref, V_val; scaled=true)
    estimated_d_geom = argmin(scaled_risk_curve)

    # 5. Plot Risk Configuration
    plot_risk_curve!(ax_risk, scaled_risk_curve, d_true; label="α = $α (d̂ = $estimated_d_geom)", color=c)
end

axislegend(ax_scree, position=:rt)
axislegend(ax_risk, position=:lt)

display(fig)

# Save the Figure
filename = "figures/vary_split_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)
