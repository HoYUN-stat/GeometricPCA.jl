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
n = 300
γ = 0.2
d_true = 15
s = 1.0
D_ref = 20
α = 0.7 # Ratio of data to use for the training/reference fold

p = round(Int, γ * n)
#λ_l = 1 + √γ * s * (d_true-l)
lambdas = collect(d_true:-1:1) .* s .* sqrt(γ) .+ 1.0

# Sanity check : Expected minimum
lower_idx(lambdas, γ; α)

# --------------------------------------------------------------------
#          Data Generation
# --------------------------------------------------------------------
Random.seed!(123);
# X = generate_spiked_data(n, p, d_true, λ);
X = generate_spiked_data(n, p, lambdas);

# Split Data for Geometric Risk
Random.seed!(123);
idx = randperm(n);
split_point = round(Int, n * α)
X_train = view(X, idx[1:split_point], :);
X_val = view(X, idx[split_point+1:end], :);
println("Train fold size: $(size(X_train)), Validation fold size: $(size(X_val))")

# Compute SVD on Folds
_, _, U_reference = svd(X_train);
U_ref = SemiOrthogonal(U_reference[:, 1:D_ref]);

_, _, V_validate = svd(X_val);
V_val = SemiOrthogonal(V_validate[:, 1:D_ref]);

# Evaluate Geometric Risk
# risk_curve = geometric_risk_sequential(U_ref, V_val)
scaled_risk_curve = geometric_risk_sequential(U_ref, V_val; scaled=true)

# Display Geometric Risk Results
estimated_d_geom = argmin(scaled_risk_curve)
println("Scaled Risk values: ", round.(scaled_risk_curve; digits=2))

# Calculate Eigenvalues for Scree Plot
# Eigenvalues from the full sample covariance matrix
full_cov = cov(X)
sample_eigenvalues = eigen(Symmetric(full_cov)).values[end:-1:end-D_ref+1]
true_eigenvalues = [lambdas; fill(1.0, p - d_true)][1:D_ref]
# println("Top $D_max sample eigenvalues: ", round.(sample_eigenvalues; digits=2))
# println("Top $D_max true eigenvalues:   ", round.(true_eigenvalues; digits=2))

# --------------------------------------------------------------------
#          Visualization
# --------------------------------------------------------------------
fig = Figure(size=(800, 600), fontsize=25)
fig[0, 1:2] = Label(fig, "PCA plot (n=$n, p=$p, D=$D_ref, d*=$d_true)", fontsize=24, tellwidth=false)

ax_risk = Axis(fig[1, 1],
    title="Geometric PCA (α = $α)",
    xlabel="Candidate Dimension (k)",
    ylabel="Scaled Chordal Risk"
)
plot_risk_curve!(ax_risk, scaled_risk_curve, d_true, p)

ax_scree = Axis(fig[1, 2],
    title="Scree Plot",
    xlabel="Principal Component Index",
    ylabel="Eigenvalue"
)
plot_scree!(ax_scree, sample_eigenvalues, true_eigenvalues, d_true)

display(fig)

# Save the Figure
filename = "figures/pca_comparison_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)
