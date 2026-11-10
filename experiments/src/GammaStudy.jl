module GammaStudy

using Random, LinearAlgebra, Statistics, Distributions, Printf, DelimitedFiles
using SHA, TOML, Serialization, Dates, Sockets
include("GDSCore.jl")
using .GDSSim
include("SpectralFrames.jl")

export StudyConfig, conditions4, run_study, parse_options, tw_quantile,
       spectral_constants, replicate, native_noise, candidate_scores, main

const EPSILONS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20]
const GAMMAS = [0.25, 0.5, 1.0, 2.0]
const TW_TABLE = readdlm(joinpath(@__DIR__, "..", "data", "tw1_table.csv"), ',', Float64; skipstart = 1)

function tw_cdf(x)
    x <= TW_TABLE[1, 1] && return TW_TABLE[1, 2]
    x >= TW_TABLE[end, 1] && return TW_TABLE[end, 2]
    i = clamp(searchsortedlast(view(TW_TABLE, :, 1), x), 1, size(TW_TABLE, 1) - 1)
    h = TW_TABLE[i + 1, 1] - TW_TABLE[i, 1]
    t = (x - TW_TABLE[i, 1]) / h
    return (2t^3 - 3t^2 + 1) * TW_TABLE[i, 2] + (t^3 - 2t^2 + t) * h * TW_TABLE[i, 3] +
           (-2t^3 + 3t^2) * TW_TABLE[i + 1, 2] + (t^3 - t^2) * h * TW_TABLE[i + 1, 3]
end

function tw_quantile(probability)
    0 < probability < 1 || throw(ArgumentError("invalid TW probability"))
    lo, hi = TW_TABLE[1, 1], TW_TABLE[end, 1]
    tw_cdf(lo) <= probability <= tw_cdf(hi) || throw(ArgumentError("probability outside supplied TW grid"))
    for _ in 1:70
        mid = (lo + hi) / 2
        if tw_cdf(mid) < probability
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

Base.@kwdef struct StudyConfig
    n::Int = 1000
    D::Int = 20
    reps::Int = 50
    gammas::Vector{Float64} = copy(GAMMAS)
    epsilons::Vector{Float64} = copy(EPSILONS)
    scenarios::Vector{Int} = [1, 2, 3, 4]
    noises::Vector{String} = String[]
    workers::Int = 2
    retry_failures::Bool = false
    phase::String = "exploration"
    seed::Int = 20261019
    basis_seed::Union{Nothing,Int} = nothing
    out::String = joinpath(@__DIR__, "..", "outputs", "runs", "exploration")
end

function conditions4(scenario)
    if scenario == 1
        return [(; tag = noise, noise, spikes = [3.6, 3.6, 3.2, 3.2], amplitude = 1.0)
                for noise in ("gaussian", "student5")]
    elseif scenario == 2
        return [(; tag = a == 1 ? "gaussian" : "mixture$a", noise = a == 1 ? "gaussian" : "mixture$a",
                  spikes = [40.0, 30.0, 25.0, 20.0], amplitude = Float64(a)) for a in (1, 10, 20, 30)]
    elseif scenario == 3
        return [(; tag = noise, noise, spikes = [12.0, 9.0, 7.0, 5.0], amplitude = 1.0)
                for noise in ("gaussian", "student5", "student8")]
    elseif scenario == 4
        names = ["gaussian", "mixture10", "mixture20", "mixture30", "student5", "student8"]
        return [(; tag = noise, noise, spikes = Float64[],
                  amplitude = startswith(noise, "mixture") ? parse(Float64, noise[8:end]) : 1.0) for noise in names]
    end
    throw(ArgumentError("scenario must be 1, 2, 3, or 4 (pure null)"))
end

function native_noise(rng, n, p, condition)
    if startswith(condition.noise, "student")
        nu = parse(Int, condition.noise[8:end])
        return rand(rng, TDist(nu), n, p) .* sqrt((nu - 2) / nu)
    end
    return noise_matrix(rng, n, p, condition.noise, condition.amplitude)
end

function spectral_constants(cfg, p)
    return (; cache = mp_cache(cfg.n, p),
            hcuts = [h_quantile(cfg.D, 1 - e) for e in cfg.epsilons],
            twcuts = [tw_quantile(1 - e) for e in cfg.epsilons])
end

function candidate_scores(q, nuclear_prefix, t)
    D = length(q)
    t > 0 || throw(ArgumentError("threshold must be positive"))
    k = collect(1:D)
    theta = vcat(0.0, 2 .* k .* t .- 2 .* cumsum(q))
    wasserstein = vcat(0.0, 2 .* k .* sqrt(t) .- 2 .* nuclear_prefix)
    return argmin(theta) - 1, argmin(wasserstein) - 1
end

function replicate(cfg, scenario, ci, condition, p, rep, U, constants)
    data_seed, split_seed, basis_seed = replicate_seeds(cfg, scenario, ci, p, rep)
    Z = native_noise(MersenneTwister(data_seed), cfg.n, p, condition)
    X = isempty(condition.spikes) ? Z : Z + (Z * U .* reshape(sqrt.(condition.spikes) .- 1, 1, :)) * U'
    order = randperm(MersenneTwister(split_seed), cfg.n)
    n1 = fld(cfg.n, 2)
    full = spectral_frame(X; lazy = true, partial = cfg.phase != "exploration")
    eigenvalues = full.S.^2 / cfg.n
    V1 = spectral_frame(X[order[1:n1], :]; leading = cfg.D).V
    V2 = spectral_frame(X[order[n1+1:end], :]; leading = cfg.D).V
    C = V1' * V2
    q = vec(sum(abs2, C; dims = 1))
    nuclear_prefix = [sum(svdvals(C[:, 1:k])) for k in 1:cfg.D]
    gd, gd_sigma, _ = gd_rank(eigenvalues, constants.cache)
    ddpa = ddpa_from_frame(eigenvalues, full, X, vec(sum(abs2, X; dims = 1)) / cfg.n, p / cfg.n)
    ranks = Tuple{String,Float64,Int,Float64}[]
    for (index, epsilon) in enumerate(cfg.epsilons)
        t = constants.hcuts[index] / p
        theta, wasserstein = candidate_scores(q, nuclear_prefix, t)
        bema, sigma2, _ = bema0_rank(eigenvalues, constants.cache; tw_quantile = constants.twcuts[index])
        push!(ranks, ("GDS-Theta", epsilon, theta, NaN))
        push!(ranks, ("GDS-W", epsilon, wasserstein, NaN))
        push!(ranks, ("BEMA0", epsilon, bema, sigma2))
    end
    append!(ranks, [("GD", NaN, gd, gd_sigma), ("DDPA", NaN, ddpa, NaN)])
    d = length(condition.spikes)
    V = right_vectors(full, X, 1:maximum(r[3] for r in ranks))
    losses = Dict(k => full_losses(U, V, k) for k in unique([r[3] for r in ranks]))
    rows = [(; phase = cfg.phase, scenario, case = condition.tag, n = cfg.n, p, gamma = p / cfg.n,
              D = cfg.D, rep, data_seed, split_seed, basis_seed, true_rank = d,
              method, epsilon, selected_rank = k, exact_rank = Int(k == d),
              underselection = Int(k < d), overselection = Int(k > d), absolute_rank_error = abs(k - d),
              chordal_loss = losses[k][1], wasserstein_loss = losses[k][2], sigma2_estimate = sigma2)
            for (method, epsilon, k, sigma2) in ranks]
    alignments = [(; phase = cfg.phase, scenario, case = condition.tag, n = cfg.n, p,
                    gamma = p / cfg.n, D = cfg.D, rep, k, q = q[k], nuclear_prefix = nuclear_prefix[k]) for k in 1:cfg.D]
    return rows, alignments
end

include("RunSupport.jl")

function parse_options(args = ARGS)
    options = Dict{String,String}()
    iseven(length(args)) || throw(ArgumentError("arguments must be option/value pairs"))
    for i in 1:2:length(args)
        args[i] in ["--n", "--D", "--reps", "--gammas", "--epsilons", "--scenarios", "--phase", "--seed", "--basis-seed", "--out", "--noises", "--workers", "--retry-failures"] || throw(ArgumentError("unknown option $(args[i])"))
        options[args[i]] = args[i+1]
    end
    phase = get(options, "--phase", "exploration")
    cfg = StudyConfig(n = parse(Int, get(options, "--n", "1000")),
                      D = parse(Int, get(options, "--D", "20")),
                      reps = parse(Int, get(options, "--reps", phase == "exploration" ? "50" : "200")),
                      gammas = parse.(Float64, split(get(options, "--gammas", "0.25,0.5,1,2"), ',')),
                      epsilons = parse.(Float64, split(get(options, "--epsilons", "0.001,0.005,0.01,0.025,0.05,0.1,0.2"), ',')),
                      scenarios = parse.(Int, split(get(options, "--scenarios", "1,2,3,4"), ',')),
                      noises = isempty(get(options, "--noises", "")) ? String[] : split(options["--noises"], ','),
                      workers = parse(Int, get(options, "--workers", "2")),
                      retry_failures = parse(Bool, get(options, "--retry-failures", "false")),
                      phase = phase, seed = parse(Int, get(options, "--seed", "20261019")),
                      basis_seed = haskey(options, "--basis-seed") ? parse(Int, options["--basis-seed"]) : nothing,
                      out = abspath(get(options, "--out", joinpath(@__DIR__, "..", "outputs", "runs", phase))))
    return cfg
end

main() = run_study(parse_options())
end
