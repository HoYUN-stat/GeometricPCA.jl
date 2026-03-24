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
γ = 1.0
d_true = 15
D_ref = 20
α = 0.5  # Fixed split ratio

ns = [50, 100, 200, 400, 800]  # Varying sample sizes
s = 4.0         # Slope
poly_deg = 1.0    # Degree of Polynomial
lambdas = collect(d_true:-1:1) .^ (poly_deg) .* s .* sqrt(γ) .+ 1.0;

limit_scaled_risk = [limit_geometric_risk(lambdas, γ, D_ref, k; α=α) for k in 1:D_ref] ./ D_ref;


# --------------------------------------------------------------------
#          Data Generation & Visualization
# --------------------------------------------------------------------
fig = Figure(size=(1000, 1000), fontsize=25)
Label(fig[0, 1], "Varying Sample Size n (γ=$γ, α=$α, D=$D_ref, d*=$d_true)", font=:bold, tellwidth=false)

ax_risk = Axis(fig[1, 1], title="Sample Risk Curves", xlabel="Candidate Dimension (k)", ylabel="Scaled Chordal Risk")

colors = [:magenta, :blue, :green, :purple, :orange]

for (i, n) in enumerate(ns)
    c = colors[i]

    p = round(Int, γ * n)

    # 1. Generate Data
    Random.seed!(123)
    X = generate_spiked_data(n, p, lambdas)

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
    plot_risk_curve!(ax_risk, scaled_risk_curve, d_true; label="n = $n (d̂ = $estimated_d_geom)", color=c)
end

plot_risk_curve!(ax_risk, limit_scaled_risk, d_true; label="Asymptotic", color=:black, linestyle=:dot)


axislegend(ax_risk, position=:lt)

display(fig)

# Save the Figure
filename = "figures/vary_sampsize_$(Dates.format(now(), "yymmdd_HHMMSS")).png"
save(filename, fig)
