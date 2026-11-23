# cd("experiments")
# ] activate .
# dev ..
using GeometricPCA
using LinearAlgebra
using Random
using Statistics
using Dates
using CairoMakie

include("../utils/plot_utils.jl")
using .PlotUtils

# --- Main Script Execution ---

# --------------------------------------------------------------------
#          Global Parameters
# --------------------------------------------------------------------
n = 100
γ = 10.0
d_true = 15
D_ref = 20
α = 0.5 # Ratio of data to use for the training/reference fold

p = round(Int, γ * n)
thres_lambda = sqrt(γ) * signal_thres(γ; α) + 1.0

# Polynomial Decay
s = 3.0         # Slope
poly_deg = 1.5    # Degree of Polynomial
lambdas = collect(d_true:-1:1) .^ (poly_deg) .* s .* sqrt(γ) .+ 1.0;


# Sanity check : Expected minimum index
limit_d_geom = lower_idx(lambdas, γ; α)
limit_scaled_risk = [limit_geometric_risk(lambdas, γ, D_ref, k; p=2, α=α) for k in 1:D_ref] ./ D_ref;
limit_scaled_risk_BW = [limit_geometric_risk(lambdas, γ, D_ref, k; p=1, α=α) for k in 1:D_ref] ./ D_ref;


# --------------------------------------------------------------------
#          Data Generation
# --------------------------------------------------------------------
Random.seed!(123);
X = generate_spiked_data(n, p, lambdas);

# Split Data for Geometric Risk
Random.seed!(123);
idx = randperm(n);
split_point = round(Int, n * α)
X_train = view(X, idx[1:split_point], :);
X_val = view(X, idx[split_point+1:end], :);

# Compute SVD on Folds
_, _, U_reference = svd(X_train);
U_ref = SemiOrthogonal(U_reference[:, 1:D_ref]);

_, _, V_validate = svd(X_val);
V_val = SemiOrthogonal(V_validate[:, 1:D_ref]);

# Evaluate Geometric Risk
scaled_risk_curve = geometric_risk_sequential(U_ref, V_val; scaled=true);
scaled_risk_curve_BW = geometric_risk_sequential(U_ref, V_val; p=1, scaled=true);



# Display Geometric Risk Results
estimated_d_geom = argmin(scaled_risk_curve)
estimated_d_geom_BW = argmin(scaled_risk_curve_BW)


# Calculate Eigenvalues for Scree Plot
# Eigenvalues from the full sample covariance matrix
full_cov = cov(X);
sample_eigenvalues = eigen(Symmetric(full_cov)).values[end:-1:end-D_ref+1];
true_eigenvalues = [lambdas; fill(1.0, p - d_true)][1:D_ref];

# --------------------------------------------------------------------
#          Visualization
# --------------------------------------------------------------------

#Figure 1(a)
fig_scree= Figure(size = (500,500), fontsize=25)
ax_scree = Axis(fig_scree[1,1],
                #title="Scree Plot",
                xlabel="Principal Component Index",
                ylabel="Eigenvalue"
                )
plot_scree!(ax_scree, sample_eigenvalues, d_true; label="Sample", color=:black)
axislegend(ax_scree, position=:rt)

display(fig_scree)
#save("figures/bad_screeplot.png", fig_scree)

#Figure 1(b)
fig_pca=Figure(size = (500,500), fontsize=25)
ax_risk = Axis(fig_pca[1,1],
               #title="Geometric PCA (α = $α)",
               xlabel="Candidate Dimension (k)",
               ylabel="Scaled Chordal Risk"
               )

plot_risk_curve!(ax_risk, scaled_risk_curve, d_true; label="Chordal (d̂ = $estimated_d_geom)", color=:blue)

plot_risk_curve!(ax_risk, scaled_risk_curve_BW, d_true; label="BW (d̂ = $estimated_d_geom_BW)", color=:orange)
axislegend(ax_risk, position=:lt)
display(fig_pca)
#save("figures/risk-curve_illustration.png", fig_pca)
