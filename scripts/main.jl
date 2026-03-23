using Pkg
Pkg.activate(".")
using GeometricPCA
using CairoMakie, Random, Statistics, Dates, LinearAlgebra

# --- Main Script Execution ---

# 1. Configure Simulation
println("--- Step 1: Configuring Simulation ---")
n = 500
p = 60
true_d = 15
D_max = 40
signal_strength = 50.0
noise_var = 1.0
train_split_ratio = 0.5 # Ratio of data to use for the training/reference fold
println("Parameters: n=$n, p=$p, true_d=$true_d, D_max=$D_max, train_split_ratio=$train_split_ratio")


# 2. Generate Data
println("\n--- Step 2: Generating Spiked Covariance Data ---")
X = generate_spiked_data(n, p, true_d, signal_strength, noise_var)
println("Data matrix size: ", size(X))

# 3. Split Data for Geometric Risk
println("\n--- Step 3: Splitting Data for Geometric Risk Method ---")
idx = randperm(n)
split_point = round(Int, n * train_split_ratio)
X_train = view(X, idx[1:split_point], :)
X_val = view(X, idx[split_point+1:end], :)
println("Train fold size: $(size(X_train)), Validation fold size: $(size(X_val))")

# 4. & 5. Compute PCA on Folds
println("\n--- Step 4 & 5: Computing PCA on Folds ---")
Cov_train = cov(X_train);
eig_train = eigen(Symmetric(Cov_train));
V_reference = eig_train.vectors[:, end:-1:end-D_max+1]

Cov_val = cov(X_val);
eig_val = eigen(Symmetric(Cov_val));
U_candidates = eig_val.vectors[:, end:-1:end-D_max+1]

# 6. Evaluate Geometric Risk
println("\n--- Step 6: Evaluating Geometric Risk ---")
risk_curve = compute_geometric_risk_sequential(U_candidates, V_reference, D_max)

# 7. Display Geometric Risk Results
println("\n--- Step 7: Geometric Risk Results ---")
estimated_d_geom = argmin(risk_curve)
println("Risk values: ", round.(risk_curve; digits=2))
println("-----------------")
println("True Dimension:              ", true_d)
println("Estimated Dimension (Geom.): ", estimated_d_geom)

# 8. Calculate Eigenvalues for Scree Plot
println("\n--- Step 8: Calculating Eigenvalues for Scree Plot ---")
# Eigenvalues from the full sample covariance matrix
full_cov = cov(X)
sample_eigenvalues = sort(eigen(Symmetric(full_cov)).values, rev=true)[1:D_max]
# Theoretical true eigenvalues of the spiked model
true_eigenvalues = [fill(signal_strength + noise_var, true_d); fill(noise_var, p - true_d)][1:D_max]
println("Top $D_max sample eigenvalues: ", round.(sample_eigenvalues; digits=2))
println("Top $D_max true eigenvalues:   ", round.(true_eigenvalues; digits=2))

# 9. Plot Side-by-Side Comparison
println("\n--- Step 9: Plotting Comparison ---")
fig = Figure(size=(800, 600), fontsize=25)
fig[0, 1] = Label(fig, "PCA plot (n=$n, p=$p, D=$D_max, d_true=$true_d, SNR=$signal_strength)", fontsize=24, tellwidth=false)

ax_risk = Axis(fig[1, 1],
    title="Geometric Risk Method (Split Ratio: $train_split_ratio)",
    xlabel="Candidate Dimension (k)",
    ylabel="Squared Chordal Risk"
)
plot_risk_curve!(ax_risk, risk_curve, true_d)

#ax_scree = Axis(fig[1, 2],
    #title="Scree Plot Method (Full Dataset)",
    #xlabel="Principal Component Index",
    #ylabel="Eigenvalue"
#)
#plot_scree!(ax_scree, sample_eigenvalues, true_eigenvalues, true_d)

display(fig)

# 10. Save the Figure
#println("\n--- Step 10: Saving Figure ---")
#if !isdir("figures")
    #println("Creating 'figures' directory.")
    #mkdir("figures")
#end
#filename = "figures/pca_comparison_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
#save(filename, fig)
