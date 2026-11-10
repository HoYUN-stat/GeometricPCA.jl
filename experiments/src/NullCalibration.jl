module NullCalibration

using Random, LinearAlgebra, Statistics, Dates, SHA, TOML, Serialization, Printf
include("GammaStudy.jl")
using .GammaStudy
using .GammaStudy.GDSSim: h_quantile, write_rows

export NullConfig, haar_overlap, gaussian_overlap, overlap_statistics, null_seed,
       run_null, parse_null_options, main

Base.@kwdef struct NullConfig
    n::Int = 1000
    D::Int = 20
    gammas::Vector{Float64} = copy(GammaStudy.GAMMAS)
    epsilons::Vector{Float64} = copy(GammaStudy.EPSILONS)
    reps::Int = 100
    data_reps::Int = 10
    rep_start::Int = 1
    seed::Int = 20261019
    out::String = joinpath(@__DIR__, "..", "outputs", "null_calibration")
end

"Exact finite-p overlap: fix the reference frame by rotation, draw one thin Haar frame."
function haar_overlap(rng, p::Int, D::Int)
    1 <= D <= p || throw(ArgumentError("require 1 <= D <= p"))
    factor = qr!(randn(rng, p, D))
    Q = Matrix(factor.Q) # p by D, never a p by p orthogonal matrix
    # Positive R diagonal makes Q itself Haar; GDS is also invariant to these signs.
    Q .*= reshape([r < 0 ? -1.0 : 1.0 for r in diag(factor.R)], 1, :)
    return Q[1:D, :]
end

"The same overlap from two independent uncentered Gaussian data splits."
function gaussian_overlap(data_rng, split_rng, n::Int, p::Int, D::Int)
    D <= min(fld(n, 2), p) || throw(ArgumentError("D exceeds split rank"))
    X = randn(data_rng, n, p)
    order = randperm(split_rng, n)
    n1 = fld(n, 2)
    A = GammaStudy.spectral_frame(X[order[1:n1], :]; leading = D).V
    B = GammaStudy.spectral_frame(X[order[n1+1:end], :]; leading = D).V
    return A' * B
end

function overlap_statistics(C)
    D = size(C, 2)
    q = vec(sum(abs2, C; dims = 1))
    nuclear = [sum(svdvals(C[:, 1:k])) for k in 1:D]
    all(isfinite, q) && all(isfinite, nuclear) || error("nonfinite frame overlap")
    maximum(q) <= 1 + 1e-8 || error("frame overlap exceeds one")
    return (; q, nuclear, max_prefix_mean = maximum(cumsum(q) ./ (1:D)),
            max_nuclear_mean = maximum(nuclear ./ (1:D)))
end

function null_seed(cfg, generator, p, rep)
    generator in ("haar", "gaussian_data") || throw(ArgumentError("unknown generator"))
    0 <= rep < 1_000_000 || throw(ArgumentError("rep must lie in 0:999999"))
    1 <= p < 100_000_000 || throw(ArgumentError("p outside disjoint seed range"))
    namespace = generator == "haar" ? 4 : 5
    return cfg.seed + namespace * 10^15 + p * 10^6 + rep
end

function validate(cfg)
    cfg.n >= 10 && iseven(cfg.n) && cfg.D > 0 || throw(ArgumentError("positive D and even n >= 10 required"))
    0 < cfg.reps < 1_000_000 || throw(ArgumentError("require 0 < reps < 1000000"))
    0 <= cfg.data_reps < 1_000_000 || throw(ArgumentError("invalid data-reps"))
    1 <= cfg.rep_start && cfg.rep_start + max(cfg.reps, cfg.data_reps) - 1 < 1_000_000 ||
        throw(ArgumentError("retained replicate IDs must lie in 1:999999; warm-up always uses 0"))
    0 <= cfg.seed < 10^12 || throw(ArgumentError("seed must lie in 0:999999999999"))
    !isempty(cfg.gammas) && all(g -> isfinite(g) && g > 0 && isinteger(cfg.n * g), cfg.gammas) || throw(ArgumentError("gammas must give positive integer p=n*gamma"))
    ps = round.(Int, cfg.n .* cfg.gammas)
    length(unique(ps)) == length(ps) || throw(ArgumentError("duplicate p values"))
    !isempty(cfg.epsilons) && all(e -> 0 < e < 1, cfg.epsilons) || throw(ArgumentError("invalid epsilon grid"))
    length(unique(cfg.epsilons)) == length(cfg.epsilons) || throw(ArgumentError("duplicate epsilon values"))
    for p in ps
        cfg.D <= min(p, fld(cfg.n, 2)) || throw(ArgumentError("D exceeds split rank"))
        null_seed(cfg, "haar", p, 0)
        all(e -> h_quantile(cfg.D, 1 - e) < p, cfg.epsilons) || throw(ArgumentError("H_D quantile must be below p"))
    end
end

function atomic_write(f, path)
    GammaStudy.atomic_write(f, path)
end

function write_toml(path, value)
    atomic_write(io -> TOML.print(io, value; sorted = true), path)
end

function setup_run(cfg)
    root = normpath(joinpath(@__DIR__, ".."))
    out = abspath(cfg.out)
    GammaStudy.check_output_path(out)
    sources = ["src/NullCalibration.jl", "src/GammaStudy.jl", "src/GDSCore.jl",
               "src/SpectralFrames.jl", "src/RunSupport.jl", "run_null.jl", "Project.toml", "Manifest.toml",
               "study_plan.json", "epsilon_lock.json", "data/tw1_table.csv"]
    hashes = Dict(file => bytes2hex(sha256(read(joinpath(root, file)))) for file in sources)
    config = Dict("kind" => "GDS Gaussian-null calibration", "n" => cfg.n, "D" => cfg.D,
                  "gammas" => cfg.gammas, "p" => round.(Int, cfg.n .* cfg.gammas),
                  "epsilons" => cfg.epsilons, "haar_reps" => cfg.reps, "data_reps" => cfg.data_reps,
                  "rep_start" => cfg.rep_start, "haar_rep_last" => cfg.rep_start + cfg.reps - 1,
                  "data_rep_last" => cfg.data_reps == 0 ? 0 : cfg.rep_start + cfg.data_reps - 1,
                  "warmup_rep" => 0,
                  "master_seed" => cfg.seed, "schema_version" => 1,
                  "seed_rule" => "master_seed + generator_namespace*10^15 + p*10^6 + rep; Haar=4, data=5; split=data+5*10^14",
                  "source_sha256" => hashes, "julia_version" => string(VERSION), "architecture" => string(Sys.ARCH))
    canonical = sprint(io -> TOML.print(io, config; sorted = true))
    identity = bytes2hex(sha256(canonical))
    manifest_path = joinpath(out, "run_manifest.toml")
    mkpath(out)
    lockdir = GammaStudy.acquire_run_lock(out)
    try
        if isfile(manifest_path)
            TOML.parsefile(manifest_path)["configuration_id"] == identity || error("configuration/source mismatch: use a new output directory")
        else
            isempty(setdiff(readdir(out), [".run_lock"])) || error("nonempty directory without null manifest: use a new output directory")
            manifest = merge(config, Dict("configuration_id" => identity,
                         "created_at_utc" => string(now(UTC)), "julia_threads" => Threads.nthreads(),
                         "blas_threads" => BLAS.get_num_threads(), "cpu" => Sys.CPU_NAME,
                         "machine" => Sys.MACHINE, "total_memory_bytes" => Sys.total_memory(),
                         "blas_config" => sprint(show, BLAS.get_config()),
                         "scope" => "Exact joint Haar-frame Gaussian null for GDS only; no competitors or non-Gaussian claims"))
            write_toml(manifest_path, manifest)
        end
        for file in sources
            destination = joinpath(out, "source_snapshot", file)
            mkpath(dirname(destination))
            if isfile(destination)
                bytes2hex(sha256(read(destination))) == hashes[file] || error("source snapshot checksum mismatch: $file")
            else
                atomic_write(io -> write(io, read(joinpath(root, file))), destination)
            end
        end
        write_toml(joinpath(out, "sessions", "$(time_ns())_$(getpid()).toml"),
                   Dict("started_at_utc" => string(now(UTC)), "julia_threads" => Threads.nthreads(),
                        "active_workers" => 1, "blas_threads" => BLAS.get_num_threads(), "pid" => getpid()))
    catch
        rm(lockdir; recursive = true, force = true)
        rethrow()
    end
    return identity, lockdir
end

function null_rep(cfg, generator, p, rep, hcuts)
    seed = null_seed(cfg, generator, p, rep)
    split_seed = seed + 5 * 10^14
    elapsed = @elapsed begin
        C = generator == "haar" ? haar_overlap(MersenneTwister(seed), p, cfg.D) :
            gaussian_overlap(MersenneTwister(seed), MersenneTwister(split_seed), cfg.n, p, cfg.D)
        stats = overlap_statistics(C)
        rows = NamedTuple[]
        for (epsilon, hcut) in zip(cfg.epsilons, hcuts)
            theta, w = candidate_scores(stats.q, stats.nuclear, hcut / p)
            for (method, rank) in (("GDS-Theta", theta), ("GDS-W", w))
                push!(rows, (; experiment = "GDS_Gaussian_null_calibration", generator,
                             n = cfg.n, p, gamma = p / cfg.n, D = cfg.D, rep,
                             seed, split_seed = generator == "haar" ? 0 : split_seed,
                             method, epsilon, selected_rank = rank, positive_dimension = Int(rank > 0),
                             first_projected_length = stats.q[1], mean_projected_length = mean(stats.q),
                             scaled_max_prefix_mean = p * stats.max_prefix_mean,
                             scaled_max_nuclear_mean_squared = p * stats.max_nuclear_mean^2))
            end
        end
    end
    return (; rows, elapsed_seconds = elapsed)
end

function wilson(successes, n; z = 1.959963984540054)
    n == 0 && return (NaN, NaN)
    p = successes / n
    center = (p + z^2 / (2n)) / (1 + z^2 / n)
    radius = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / (1 + z^2 / n)
    return max(0.0, center - radius), min(1.0, center + radius)
end

function summarize(cfg, results, failures)
    summaries = NamedTuple[]
    for generator in ("haar", "gaussian_data"), p in round.(Int, cfg.n .* cfg.gammas),
        epsilon in cfg.epsilons, method in ("GDS-Theta", "GDS-W")
        requested = generator == "haar" ? cfg.reps : cfg.data_reps
        requested == 0 && continue
        rows = filter(r -> r.generator == generator && r.p == p && r.epsilon == epsilon && r.method == method, results)
        R = length(rows)
        selected = sum(r.positive_dimension for r in rows; init = 0)
        estimate = R == 0 ? NaN : selected / R
        lower, upper = wilson(selected, R)
        count_failed = count(f -> f.generator == generator && f.p == p, failures)
        push!(summaries, (; experiment = "GDS_Gaussian_null_calibration", generator, n = cfg.n, p,
                         gamma = p / cfg.n, D = cfg.D, epsilon, method, requested_replicates = requested,
                         successful_replicates = R, failed_replicates = count_failed,
                         positive_selections = selected, probability_positive = estimate,
                         mcse = R == 0 ? NaN : sqrt(estimate * (1 - estimate) / R),
                         wilson95_lower = lower, wilson95_upper = upper))
    end
    comparisons = NamedTuple[]
    for h in filter(r -> r.generator == "haar", summaries)
        data = filter(r -> r.generator == "gaussian_data" && r.p == h.p && r.epsilon == h.epsilon && r.method == h.method, summaries)
        isempty(data) && continue
        d = only(data)
        # Independent simulations: an MC standard error, not a paired comparison or equivalence test.
        se = sqrt(h.mcse^2 + d.mcse^2)
        difference = h.probability_positive - d.probability_positive
        # Conservative difference bound from two 97.5% exact binomial intervals.
        using_beta = GammaStudy.Distributions.Beta
        quant = GammaStudy.Distributions.quantile
        function exact_interval(k, n)
            n == 0 && return (NaN, NaN)
            return (k == 0 ? 0.0 : quant(using_beta(k, n - k + 1), 0.0125),
                    k == n ? 1.0 : quant(using_beta(k + 1, n - k), 0.9875))
        end
        hlo, hhi = exact_interval(h.positive_selections, h.successful_replicates)
        dlo, dhi = exact_interval(d.positive_selections, d.successful_replicates)
        push!(comparisons, (; p = h.p, gamma = h.gamma, epsilon = h.epsilon, method = h.method,
                            haar_replicates = h.successful_replicates, data_replicates = d.successful_replicates,
                            haar_probability = h.probability_positive, data_probability = d.probability_positive,
                            haar_minus_data = difference, independent_mcse = se,
                            conservative95_lower = hlo - dhi, conservative95_upper = hhi - dlo))
    end
    return summaries, comparisons
end

function run_null(cfg::NullConfig)
    BLAS.set_num_threads(1)
    validate(cfg)
    identity, lockdir = setup_run(cfg)
    try
        write_toml(joinpath(cfg.out, "status.toml"), Dict("configuration_id" => identity, "state" => "running", "started_at_utc" => string(now(UTC))))
        all_rows, failures, timings = NamedTuple[], NamedTuple[], NamedTuple[]
        hcuts = [h_quantile(cfg.D, 1 - e) for e in cfg.epsilons]
        calibration = [(; p, n = cfg.n, D = cfg.D, epsilon = e, H_quantile = h,
                         t = h / p, eta_theta = 1 - 2h / p, eta_w = 1 - 2sqrt(h / p))
                       for p in round.(Int, cfg.n .* cfg.gammas) for (e, h) in zip(cfg.epsilons, hcuts)]
        GammaStudy.atomic_rows(joinpath(cfg.out, "calibration.csv"), calibration)
        for p in round.(Int, cfg.n .* cfg.gammas), generator in ("haar", "gaussian_data")
            R = generator == "haar" ? cfg.reps : cfg.data_reps
            R == 0 && continue
            # Warm-up is independent of all retained replicates and excluded from timings.
            warmup = try
                @elapsed null_rep(cfg, generator, p, 0, hcuts)
            catch exception
                write_toml(joinpath(cfg.out, "warmup_failure.toml"), Dict("generator" => generator, "p" => p,
                           "seed" => null_seed(cfg, generator, p, 0), "error" => sprint(showerror, exception, catch_backtrace())))
                rethrow()
            end
            new_count, resumed_count, failed_count = 0, 0, 0
            computation_seconds = 0.0
            for rep in cfg.rep_start:(cfg.rep_start + R - 1)
                checkpoint = joinpath(cfg.out, "checkpoints", "$(generator)_p$(p)_rep$(lpad(rep, 6, '0')).jls")
                if isfile(checkpoint)
                    record = GammaStudy.checkpoint_read(checkpoint, identity, (generator, p, rep))
                    resumed_count += 1
                else
                    seed = null_seed(cfg, generator, p, rep)
                    try
                        outcome = null_rep(cfg, generator, p, rep, hcuts)
                        record = (; fingerprint = identity, key = (generator, p, rep), generator, p, rep, seed, status = "ok",
                                    rows = outcome.rows, elapsed_seconds = outcome.elapsed_seconds, error = "")
                    catch exception
                        exception isa InterruptException && rethrow()
                        record = (; fingerprint = identity, key = (generator, p, rep), generator, p, rep, seed, status = "failed",
                                    rows = NamedTuple[], elapsed_seconds = NaN,
                                    error = sprint(showerror, exception, catch_backtrace()))
                    end
                    GammaStudy.checkpoint_write(checkpoint, record)
                    new_count += 1
                end
                if record.status == "ok"
                    append!(all_rows, record.rows)
                    computation_seconds += record.elapsed_seconds
                else
                    failed_count += 1
                    push!(failures, (; generator, p, rep, seed = record.seed, status = record.status, error = record.error))
                end
            end
            push!(timings, (; generator, p, requested_replicates = R, successful_replicates = R - failed_count,
                             new_replicates = new_count, resumed_replicates = resumed_count,
                             warmup_seconds = warmup, compute_seconds = computation_seconds,
                             mean_seconds = computation_seconds / max(R - failed_count, 1),
                             julia_threads = Threads.nthreads(), blas_threads = BLAS.get_num_threads()))
            @printf("GDS null %-13s p=%4d R=%d resumed=%d failed=%d warm compute=%.4fs/rep\n",
                    generator, p, R, resumed_count, failed_count, last(timings).mean_seconds)
            flush(stdout)
        end
        summaries, comparisons = summarize(cfg, all_rows, failures)
        GammaStudy.atomic_rows(joinpath(cfg.out, "null_replicates.csv"), all_rows)
        GammaStudy.atomic_rows(joinpath(cfg.out, "null_summary.csv"), summaries)
        GammaStudy.atomic_rows(joinpath(cfg.out, "haar_vs_data.csv"), comparisons)
        GammaStudy.atomic_rows(joinpath(cfg.out, "timings.csv"), timings)
        # TOML retains multiline exception messages without malformed CSV quoting.
        write_toml(joinpath(cfg.out, "failures.toml"), Dict("failures" => [Dict(string(k) => v for (k,v) in pairs(f)) for f in failures]))
        write_toml(joinpath(cfg.out, "status.toml"), Dict("configuration_id" => identity,
                   "completed_at_utc" => string(now(UTC)), "failed_replicates" => length(failures),
                   "state" => isempty(failures) ? "completed" : "failed",
                   "successful_method_rows" => length(all_rows), "label" => "new GDS Gaussian-null calibration; separate from archived confirmation"))
        isempty(failures) || error("$(length(failures)) failed null replicates recorded in failures.toml; summaries expose reduced denominators")
        return summaries
    finally
        rm(lockdir; recursive = true, force = true)
    end
end

function parse_null_options(args = ARGS)
    iseven(length(args)) || throw(ArgumentError("arguments must be option/value pairs"))
    options = Dict{String,String}()
    for i in 1:2:length(args)
        args[i] in ("--n", "--D", "--gammas", "--epsilons", "--reps", "--data-reps", "--rep-start", "--seed", "--out") || throw(ArgumentError("unknown option $(args[i])"))
        options[args[i]] = args[i+1]
    end
    return NullConfig(n = parse(Int, get(options, "--n", "1000")), D = parse(Int, get(options, "--D", "20")),
                      gammas = parse.(Float64, split(get(options, "--gammas", "0.25,0.5,1,2"), ',')),
                      epsilons = parse.(Float64, split(get(options, "--epsilons", "0.001,0.005,0.01,0.025,0.05,0.1,0.2"), ',')),
                      reps = parse(Int, get(options, "--reps", "100")),
                      data_reps = parse(Int, get(options, "--data-reps", "10")),
                      rep_start = parse(Int, get(options, "--rep-start", "1")),
                      seed = parse(Int, get(options, "--seed", "20261019")),
                      out = abspath(get(options, "--out", joinpath(@__DIR__, "..", "outputs", "null_calibration"))))
end

main() = run_null(parse_null_options())
end
