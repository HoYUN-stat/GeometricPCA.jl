module GeometricPCA

using LinearAlgebra
using Random
using Statistics
using CairoMakie
using Dates

export generate_spiked_data,
       compute_geometric_risk_sequential,
       plot_risk_curve!,
       plot_scree!,
       risk_estimate

# --- Function Definitions ---

function generate_spiked_data(n::Int, p::Int, true_d::Int, signal_strength::Float64, noise_var::Float64)
    Z = randn(p, true_d)
    V_true = Matrix(qr(Z).Q)
    Signals = randn(n, true_d) .* sqrt(signal_strength)
    X = Signals * V_true'
    X .+= randn(n, p) .* sqrt(noise_var)
    return X
end

function compute_geometric_risk_sequential(U_candidates::AbstractMatrix{Float64}, V_reference::AbstractMatrix{Float64}, D::Int)
    risk_curve = zeros(Float64, D)
    cross_matrix_buffer = zeros(Float64, D, D)
    for k in 1:D
        W_k = view(cross_matrix_buffer, 1:k, 1:D)
        U_k = view(U_candidates, :, 1:k)
        mul!(W_k, U_k', V_reference)
        sq_norm = sum(abs2, W_k)
        risk_curve[k] = (k + D - 2.0 * sq_norm) / D
    end
    return risk_curve
end

# --- Plotting Functions ---

function plot_risk_curve!(ax, risk_curve::Vector{Float64}, true_d::Int)
    D_max = length(risk_curve)
    estimated_d = argmin(risk_curve)

    lines!(ax, 1:D_max, risk_curve, linewidth=3, color=:royalblue, label="Risk Curve")
    scatter!(ax, 1:D_max, risk_curve, markersize=12, color=:royalblue)
    vlines!(ax, [true_d], color=:black, linestyle=:dash, linewidth=2, label="d_true = $true_d")
    scatter!(ax, [estimated_d], [risk_curve[estimated_d]],
        color=:red, marker=:star5, markersize=20, label="d_est = $estimated_d"
    )
    axislegend(ax, position=:rb)
end

function plot_scree!(ax, sample_eigenvalues::Vector{Float64}, true_eigenvalues::Vector{Float64}, true_d::Int)
    num_eigenvalues = length(sample_eigenvalues)
    # ax.yscale = log10

    # Clip non-positive sample eigenvalues to avoid log(0) error
    clipped_sample_eigs = max.(sample_eigenvalues, eps(Float64))

    lines!(ax, 1:num_eigenvalues, clipped_sample_eigs, linewidth=3, color=:darkorange, label="Sample Eigenvalues")
    scatter!(ax, 1:num_eigenvalues, clipped_sample_eigs, markersize=12, color=:darkorange)

    # Plot the true eigenvalues
    lines!(ax, 1:num_eigenvalues, true_eigenvalues, linewidth=2, linestyle=:dot, color=:green, label="True Eigenvalues")

    vlines!(ax, [true_d], color=:black, linestyle=:dash, linewidth=2, label="d_true = $true_d")

    axislegend(ax, position=:rt)
end

function risk_estimate(n::Int64, p::Int64, true_d::Int64, D_max::Int64, train_split_ratio, snr::Float64, noise_var::Float64)
    # Generate Data
    println("\n--- Step 2: Generating Spiked Covariance Data ---")
    X = generate_spiked_data(n, p, true_d, snr, noise_var)
    println("Data matrix size: ", size(X))

    # Split Data for Geometric Risk
    println("\n--- Step 3: Splitting Data for Geometric Risk Method ---")
    idx = randperm(n)
    split_point = round(Int, n * train_split_ratio)
    X_train = view(X, idx[1:split_point], :)
    X_val = view(X, idx[split_point+1:end], :)
    println("Train fold size: $(size(X_train)), Validation fold size: $(size(X_val))")

    # Compute PCA on Folds
    println("\n--- Step 4 & 5: Computing PCA on Folds ---")
    Cov_train = cov(X_train);
    eig_train = eigen(Symmetric(Cov_train));
    V_reference = eig_train.vectors[:, end:-1:end-D_max+1]

    Cov_val = cov(X_val);
    eig_val = eigen(Symmetric(Cov_val));
    U_candidates = eig_val.vectors[:, end:-1:end-D_max+1]

    # Evaluate Geometric Risk
    println("\n--- Step 6: Evaluating Geometric Risk ---")
    risk_curve = compute_geometric_risk_sequential(U_candidates, V_reference, D_max)

    # Display Geometric Risk Results
    println("\n--- Step 7: Geometric Risk Results ---")
    estimated_d_geom = argmin(risk_curve)
    println("Risk values: ", round.(risk_curve; digits=2))
    println("-----------------")
    println("True Dimension:              ", true_d)
    println("Estimated Dimension (Geom.): ", estimated_d_geom)
    return (risk_curve)
end

end
