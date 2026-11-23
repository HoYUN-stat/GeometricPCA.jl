module GDSSim

using LinearAlgebra, Statistics, Random, Printf, Dates
using Distributions, QuadGK

export h_cdf, h_quantile, mp_quantile, mp_cache, upper_edge, ddpa_rank,
       gds_scores, select_smallest, bema0_rank, gd_rank, make_data,
       Config, conditions, run_scenario, parse_cli, main, METHODS, single_rep,
       basis_for, noise_matrix, write_rows, full_losses

const METHODS = ["GDS-Theta", "GDS-W", "BEMA0", "GD", "DDPA"]

"CDF of max_{1 <= k <= D} mean(Y[1:k]), with independent Y[j] ~ Chisq(D)."
function h_cdf(x::Real, D::Integer)
    D >= 1 || throw(ArgumentError("D must be positive"))
    x <= 0 && return 0.0
    probabilities = [cdf(Chisq(l * D), l * x) for l in 1:D]
    r = zeros(D + 1)
    r[1] = 1.0
    for m in 1:D
        r[m + 1] = sum(probabilities[l] * r[m - l + 1] for l in 1:m) / m
    end
    return clamp(r[end], 0.0, 1.0)
end

function h_quantile(D::Integer, probability::Real = 0.95)
    0 < probability < 1 || throw(ArgumentError("probability must lie in (0, 1)"))
    lo, hi = 0.0, max(2.0 * D, 1.0)
    while h_cdf(hi, D) < probability
        hi *= 2
    end
    for _ in 1:70
        mid = (lo + hi) / 2
        if h_cdf(mid, D) < probability
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

"Unit-scale MP quantile for aspect ratio beta <= 1 (continuous, mass one)."
function mp_quantile(probability::Real, beta::Real)
    0 < beta <= 1 || throw(ArgumentError("MP beta must lie in (0, 1]"))
    0 <= probability <= 1 || throw(ArgumentError("invalid probability"))
    probability == 0 && return (1 - sqrt(beta))^2
    probability == 1 && return (1 + sqrt(beta))^2
    # x(theta) = 1 + beta - 2 sqrt(beta) cos(theta).
    # At beta=1 the integrated transformed density has a closed form.
    function angular_cdf(theta)
        beta == 1 && return (theta + sin(theta)) / pi
        integrand(v) = 2 * sin(v)^2 / (pi * (1 + beta - 2 * sqrt(beta) * cos(v)))
        return quadgk(integrand, 0.0, theta; rtol = 1e-10)[1]
    end
    lo, hi = 0.0, Float64(pi)
    for _ in 1:60
        mid = (lo + hi) / 2
        if angular_cdf(mid) < probability
            lo = mid
        else
            hi = mid
        end
    end
    return 1 + beta - 2 * sqrt(beta) * cos((lo + hi) / 2)
end

function mp_cache(n::Int, p::Int; trim = 0.2)
    m, N = min(n, p), max(n, p)
    beta = m / N
    indices = max(1, floor(Int, m * trim)):min(m, floor(Int, m * (1 - trim)))
    # Match the pairing of reversed sample eigenvalues and lower MP quantiles
    # in the authors' BEMA0.R. The size choices used here make the endpoints exact.
    predictor = [mp_quantile(k / m, beta) * N / n for k in indices]
    return (; n, p, m, N, beta, indices, predictor,
            mp_median = mp_quantile(0.5, beta) * N / n)
end

"BEMA0: isotropic bulk matching; beta_error=0.05 uses TW1 0.95 quantile 0.9793."
function bema0_rank(eigenvalues, cache; tw_quantile = 0.9793)
    sample_bulk = reverse(eigenvalues[cache.indices])
    sigma2 = dot(sample_bulk, cache.predictor) / sum(abs2, cache.predictor)
    beta, N = cache.beta, cache.N
    edge = (1 + sqrt(beta))^2
    correction = tw_quantile * N^(-2 / 3) * beta^(-1 / 6) * (1 + sqrt(beta))^(4 / 3)
    threshold = sigma2 * (N / cache.n) * (edge + correction)
    return count(>(threshold), eigenvalues), sigma2, threshold
end

"Gavish--Donoho hard threshold with unknown variance estimated by the MP median."
function gd_rank(eigenvalues, cache)
    beta = cache.beta
    sigma2 = median(eigenvalues) / cache.mp_median
    multiplier = 2 * (1 + beta) + 8 * beta / (1 + beta + sqrt(beta^2 + 14 * beta + 1))
    threshold = sigma2 * (cache.N / cache.n) * multiplier
    return count(>(threshold), eigenvalues), sigma2, threshold
end

"Upper MP-deformed edge for an empirical distribution of diagonal variances."
function upper_edge(diagonal_variances, gamma::Real)
    gamma > 0 || throw(ArgumentError("gamma must be positive"))
    maximum_variance = maximum(diagonal_variances)
    maximum_variance <= 0 && return 0.0
    normalized = diagonal_variances ./ maximum_variance
    lo, hi = 0.0, 1.0
    # Solve the edge equation on the rightmost branch. Keeping all p variances
    # avoids the 100-quantile numerical thinning in the authors' MATLAB helper.
    for _ in 1:70
        u = (lo + hi) / 2
        derivative_term = gamma * mean((u .* normalized ./ (1 .- u .* normalized)).^2)
        if derivative_term < 1
            lo = u
        else
            hi = u
        end
    end
    u = (lo + hi) / 2
    return maximum_variance * (1 / u + gamma * mean(normalized ./ (1 .- u .* normalized)))
end

"Plain DDPA, corresponding to ddpa(X, 1, 0), not DDPA+."
function ddpa_rank(eigenvalues, V, column_variances, gamma; inflation = 0.0)
    residual_variances = copy(column_variances)
    selected, last_edge = 0, NaN
    for j in eachindex(eigenvalues)
        last_edge = upper_edge(residual_variances, gamma)
        if eigenvalues[j] > last_edge * (1 + inflation)^2
            selected += 1
            residual_variances .-= eigenvalues[j] .* abs2.(view(V, :, j))
            residual_variances .= max.(residual_variances, 0.0)
        else
            break
        end
    end
    return selected, last_edge
end

select_smallest(values) = argmin(values) - 1

function gds_scores(C::AbstractMatrix, t::Real)
    D = size(C, 1)
    size(C, 2) == D || throw(ArgumentError("C must be D by D"))
    0 < t < 1 || throw(ArgumentError("H_D quantile / p must be in (0, 1)"))
    q = vec(sum(abs2, C; dims = 1))
    theta = vcat(Float64(D), D .+ 2 .* collect(1:D) .* t .- 2 .* cumsum(q))
    wasserstein = [k == 0 ? Float64(D) : D + 2 * k * sqrt(t) - 2 * sum(svdvals(C[:, 1:k])) for k in 0:D]
    return (; theta, wasserstein, q, rank_theta = select_smallest(theta),
            rank_w = select_smallest(wasserstein))
end

Base.@kwdef struct Config
    reps::Int = 200
    sizes::Vector{Int} = [500, 1000]
    D::Int = 20
    epsilon::Float64 = 0.05
    seed::Int = 20261019
    out::String = joinpath(@__DIR__, "..", "results")
end

function conditions(scenario::Int)
    if scenario == 1
        return [(; noise, spikes = [3.6, 3.6, 3.2, 3.2], amplitude = 1.0)
                for noise in ("gaussian", "rademacher", "bernoulli01")]
    elseif scenario == 2
        return [(; noise = a == 1 ? "gaussian" : "mixture$(a)",
                  spikes = [40.0, 30.0, 25.0, 20.0], amplitude = Float64(a))
                for a in (1, 10, 20, 30)]
    elseif scenario == 3
        return [(; noise, spikes = [12.0, 9.0, 7.0, 5.0], amplitude = 1.0)
                for noise in ("gaussian", "rademacher", "bernoulli01")]
    elseif scenario == 0
        names = ["gaussian", "rademacher", "bernoulli01", "mixture10", "mixture20", "mixture30"]
        return [(; noise = name, spikes = Float64[],
                  amplitude = startswith(name, "mixture") ? parse(Float64, name[8:end]) : 1.0)
                for name in names]
    else
        throw(ArgumentError("scenario must be 0, 1, 2, or 3"))
    end
end

function noise_matrix(rng, n, p, noise, amplitude)
    if noise == "gaussian"
        return randn(rng, n, p)
    elseif noise == "rademacher"
        return 2.0 .* rand(rng, Bool, n, p) .- 1.0
    elseif noise == "bernoulli01"
        return (Float64.(rand(rng, n, p) .< 0.1) .- 0.1) ./ sqrt(0.09)
    elseif startswith(noise, "mixture")
        Z = randn(rng, n, p)
        indicators = rand(rng, n, p) .< 1e-4
        Z .*= 1 .+ (amplitude - 1) .* indicators
        Z ./= sqrt(1 + 1e-4 * (amplitude^2 - 1))
        return Z
    end
    throw(ArgumentError("unknown noise distribution"))
end

function basis_for(seed, p, d)
    d == 0 && return zeros(p, 0)
    rng = MersenneTwister(seed)
    return Matrix(qr(randn(rng, p, d)).Q)[:, 1:d]
end

function make_data(rng, n, p, condition, U)
    Z = noise_matrix(rng, n, p, condition.noise, condition.amplitude)
    isempty(condition.spikes) && return Z
    return Z + (Z * U .* reshape(sqrt.(condition.spikes) .- 1, 1, :)) * U'
end

function full_losses(U, V, k)
    d = size(U, 2)
    if d == 0 || k == 0
        return Float64(d + k), Float64(d + k)
    end
    overlap = U' * V[:, 1:k]
    return d + k - 2 * sum(abs2, overlap), d + k - 2 * sum(svdvals(overlap))
end

function single_rep(scenario, condition, ci, n, cfg, rep, U, cache, hcut)
    p = n
    # No use of Julia's session-dependent hash: these seeds are identical across
    # thread counts, machine scheduling, and separate scenario entry points.
    data_seed = cfg.seed + 100_000_000 * scenario + 1_000_000 * ci + 1_000 * n + rep
    split_seed = data_seed + 600_000_000
    basis_seed = cfg.seed + 900_000_000 + 1_000 * n
    X = make_data(MersenneTwister(data_seed), n, p, condition, U)
    permutation = randperm(MersenneTwister(split_seed), n)
    n1 = fld(n, 2)
    full = svd(X; full = false)
    eigenvalues = full.S.^2 ./ n
    V = full.V
    split1 = svd(X[permutation[1:n1], :]; full = false)
    split2 = svd(X[permutation[n1+1:end], :]; full = false)
    C = split1.V[:, 1:cfg.D]' * split2.V[:, 1:cfg.D]
    scores = gds_scores(C, hcut / p)
    bema, bema_sigma, _ = bema0_rank(eigenvalues, cache)
    gd, gd_sigma, _ = gd_rank(eigenvalues, cache)
    ddpa, _ = ddpa_rank(eigenvalues, V, vec(sum(abs2, X; dims = 1)) ./ n, p / n)
    selected = [scores.rank_theta, scores.rank_w, bema, gd, ddpa]
    d = length(condition.spikes)
    rows = NamedTuple[]
    for (method, k) in zip(METHODS, selected)
        chordal, wasserstein = full_losses(U, V, k)
        push!(rows, (; scenario, noise = condition.noise, n, p, D = cfg.D,
                     epsilon = cfg.epsilon, rep, data_seed, split_seed, basis_seed,
                     true_rank = d, method, selected_rank = k,
                     exact_rank = Int(k == d), underselection = Int(k < d),
                     overselection = Int(k > d), absolute_rank_error = abs(k - d),
                     chordal_loss = chordal, wasserstein_loss = wasserstein,
                     sigma2_estimate = method == "BEMA0" ? bema_sigma : method == "GD" ? gd_sigma : NaN))
    end
    curve_rows = [(; scenario, noise = condition.noise, n, p, D = cfg.D,
                   epsilon = cfg.epsilon, rep, k,
                   theta = scores.theta[k + 1], wasserstein = scores.wasserstein[k + 1],
                   q = k == 0 ? NaN : scores.q[k]) for k in 0:cfg.D]
    return rows, curve_rows
end

function write_rows(path, rows)
    isempty(rows) && return
    mkpath(dirname(path))
    # All string fields are fixed identifiers without commas or newlines.
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            println(io, join(string.(values(row)), ','))
        end
    end
end

function run_scenario(scenario::Int, cfg::Config)
    cfg.epsilon == 0.05 || throw(ArgumentError("This comparator setup fixes BEMA's TW1 .95 quantile; use epsilon=0.05 or explicitly update BEMA tuning."))
    cfg.reps > 0 || throw(ArgumentError("reps must be positive"))
    BLAS.set_num_threads(1)
    hcut = h_quantile(cfg.D, 1 - cfg.epsilon)
    mkpath(cfg.out)
    println("Julia $(VERSION); threads=$(Threads.nthreads()); BLAS threads=$(BLAS.get_num_threads())")
    println("Scenario $scenario; seed=$(cfg.seed); repetitions=$(cfg.reps); H quantile=$hcut")
    flush(stdout)
    for n in cfg.sizes
        cfg.D < n && cfg.D <= fld(n, 2) || throw(ArgumentError("D must be < p and <= both split ranks"))
        hcut < n || throw(ArgumentError("H_D quantile must be smaller than p"))
        cache = mp_cache(n, n)
        for (ci, condition) in enumerate(conditions(scenario))
            U = basis_for(cfg.seed + 900_000_000 + 1_000 * n, n, length(condition.spikes))
            results = Vector{Any}(undef, cfg.reps)
            elapsed = @elapsed Threads.@threads :dynamic for rep in 1:cfg.reps
                results[rep] = single_rep(scenario, condition, ci, n, cfg, rep, U, cache, hcut)
            end
            rows = reduce(vcat, (result[1] for result in results))
            curve_rows = reduce(vcat, (result[2] for result in results))
            stem = "scenario$(scenario)_$(condition.noise)_n$(n)"
            write_rows(joinpath(cfg.out, stem * "_replicates.csv"), rows)
            write_rows(joinpath(cfg.out, stem * "_curves.csv"), curve_rows)
            @printf("Completed %-35s %4d replicates in %7.2f s\n", stem, cfg.reps, elapsed)
            for method in METHODS
                method_rows = filter(r -> r.method == method, rows)
                @printf("  %-10s mean rank=%6.3f  exact=%6.3f  over=%6.3f  chordal=%7.3f\n",
                        method, mean(r.selected_rank for r in method_rows),
                        mean(r.exact_rank for r in method_rows), mean(r.overselection for r in method_rows),
                        mean(r.chordal_loss for r in method_rows))
            end
            flush(stdout)
        end
    end
end

function parse_cli(args)
    options = Dict{String,String}()
    iseven(length(args)) || throw(ArgumentError("Use pairs such as --reps 200 --sizes 500,1000"))
    for i in 1:2:length(args)
        args[i] in ("--reps", "--sizes", "--D", "--epsilon", "--seed", "--out") || throw(ArgumentError("Unknown argument $(args[i])"))
        options[args[i]] = args[i+1]
    end
    return Config(reps = parse(Int, get(options, "--reps", "200")),
                  sizes = parse.(Int, split(get(options, "--sizes", "500,1000"), ',')),
                  D = parse(Int, get(options, "--D", "20")),
                  epsilon = parse(Float64, get(options, "--epsilon", "0.05")),
                  seed = parse(Int, get(options, "--seed", "20261019")),
                  out = abspath(get(options, "--out", joinpath(@__DIR__, "..", "results"))))
end

main(scenario; args = ARGS) = run_scenario(scenario, parse_cli(args))

end # module
