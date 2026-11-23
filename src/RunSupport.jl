# Execution support only. The selection and data-generation formulas remain in
# GammaStudy.jl / GDSCore.jl, with unchanged archived seed mapping.
const RUN_SCHEMA = 2

# A separate population-basis master permits fresh observation/split seeds while
# preserving the fixed population orientation from the archived study.
effective_basis_master(cfg) = isnothing(cfg.basis_seed) ? cfg.seed : cfg.basis_seed
population_basis_seed(cfg, p) = effective_basis_master(cfg) + 900_000_000 + 1_000 * p
population_basis(cfg, p, rank) = basis_for(population_basis_seed(cfg, p), p, rank)

function basis_digest(U)
    buffer = IOBuffer()
    write(buffer, "GDSBasisFloat64LE-v1 rows=$(size(U, 1)) cols=$(size(U, 2))\n")
    write(buffer, htol.(reinterpret(UInt64, vec(Matrix{Float64}(U)))))
    return bytes2hex(sha256(take!(buffer)))
end

function replicate_seeds(cfg, scenario, ci, p, rep)
    phase_code = cfg.phase == "exploration" ? 1 : cfg.phase == "confirmation" ? 2 :
                 cfg.phase == "validation" ? 3 : throw(ArgumentError("unknown phase"))
    data_seed = cfg.seed + phase_code * 10^15 + scenario * 10^12 + ci * 10^10 + p * 10^5 + rep
    return data_seed, data_seed + 5 * 10^14, population_basis_seed(cfg, p)
end

function validate_config(cfg)
    cfg.phase in ("exploration", "confirmation", "validation") || throw(ArgumentError("unknown phase"))
    cfg.n >= 10 && iseven(cfg.n) || throw(ArgumentError("n must be even and at least 10"))
    1 <= cfg.reps < 100_000 || throw(ArgumentError("reps must be 1:99999 to preserve distinct seed ranges"))
    0 <= cfg.seed <= 10^9 || throw(ArgumentError("master seed must be in 0:1000000000"))
    0 <= effective_basis_master(cfg) <= 10^9 || throw(ArgumentError("basis master seed must be in 0:1000000000"))
    1 <= cfg.workers <= 4 || throw(ArgumentError("use 1 to 4 workers; default 2 avoids memory pressure"))
    for (name, values) in (("gammas", cfg.gammas), ("epsilons", cfg.epsilons), ("scenarios", cfg.scenarios))
        !isempty(values) && length(unique(values)) == length(values) || throw(ArgumentError("$name must be nonempty and unique"))
    end
    all(s -> s in 1:4, cfg.scenarios) || throw(ArgumentError("scenarios must be 1:4"))
    all(e -> isfinite(e) && 0 < e < 1, cfg.epsilons) || throw(ArgumentError("epsilons must lie in (0,1)"))
    all(g -> isfinite(g) && g > 0 && isinteger(cfg.n * g), cfg.gammas) || throw(ArgumentError("gammas must give positive integer p=n*gamma"))
    ps = Int.(cfg.n .* cfg.gammas)
    all(p -> cfg.D >= 1 && cfg.D < p < 100_000 && cfg.D <= cfg.n ÷ 2, ps) ||
        throw(ArgumentError("require 1 <= D < p < 100000 and D <= n/2"))
    all(p -> p >= maximum(length(c.spikes) for s in cfg.scenarios for c in conditions4(s)), ps) ||
        throw(ArgumentError("p must be at least the population signal rank"))
    valid_noises = unique(c.noise for s in cfg.scenarios for c in conditions4(s))
    all(n -> n in valid_noises, cfg.noises) || throw(ArgumentError("unknown or unavailable --noises value"))
    length(unique(cfg.noises)) == length(cfg.noises) || throw(ArgumentError("duplicate noise filters"))
    return nothing
end

selected_conditions(cfg, scenario) = [(ci, c) for (ci, c) in enumerate(conditions4(scenario)) if isempty(cfg.noises) || c.noise in cfg.noises]

function source_hashes()
    root = normpath(joinpath(@__DIR__, ".."))
    files = ["src/GDSCore.jl", "src/GammaStudy.jl", "src/SpectralFrames.jl", "src/RunSupport.jl",
             "run_study.jl", "Project.toml", "Manifest.toml", "study_plan.json", "epsilon_lock.json", "data/tw1_table.csv"]
    append!(files, ["scenarios/S$s.jl" for s in 1:4])
    return Dict(f => bytes2hex(sha256(read(joinpath(root, f)))) for f in files)
end

toml_text(value) = sprint(io -> TOML.print(io, value; sorted = true))

function run_identity(cfg)
    design = Dict("n" => cfg.n, "D" => cfg.D, "reps" => cfg.reps,
                  "gammas" => cfg.gammas, "epsilons" => cfg.epsilons,
                  "scenarios" => cfg.scenarios, "noises" => cfg.noises,
                  "phase" => cfg.phase, "master_seed" => cfg.seed,
                  "basis_master_seed" => effective_basis_master(cfg))
    identity = Dict("schema" => RUN_SCHEMA, "design" => design,
                    "julia_version" => string(VERSION), "architecture" => string(Sys.ARCH),
                    "source_sha256" => source_hashes())
    return identity, bytes2hex(sha256(toml_text(identity)))
end

function atomic_write(writer, path)
    mkpath(dirname(path))
    temp, io = mktemp(dirname(path))
    try
        writer(io)
        flush(io)
        close(io)
        mv(temp, path; force = true)
    finally
        isopen(io) && close(io)
        isfile(temp) && rm(temp)
    end
end

atomic_toml(path, value) = atomic_write(io -> TOML.print(io, value; sorted = true), path)

function checkpoint_write(path, value)
    buffer = IOBuffer()
    serialize(buffer, value)
    payload = take!(buffer)
    atomic_write(path) do io
        write(io, sha256(payload))
        write(io, payload)
    end
end

function checkpoint_read(path, fingerprint, key)
    open(path) do io
        expected = read(io, 32)
        payload = read(io)
        sha256(payload) == expected || error("Corrupt checkpoint: $path. Preserve it for audit and rerun in a new output directory.")
        value = deserialize(IOBuffer(payload))
        value.fingerprint == fingerprint && value.key == key || error("Checkpoint identity mismatch: $path")
        return value
    end
end

# Resolve existing symlink ancestors before testing protected archive paths.
function resolved_path(path)
    path = abspath(path)
    ispath(path) && return realpath(path)
    return joinpath(resolved_path(dirname(path)), basename(path))
end

function check_output_path(path)
    root = realpath(joinpath(@__DIR__, ".."))
    out = resolved_path(path)
    for name in ("results", "data", "report")
        protected = joinpath(root, name)
        (out == protected || startswith(out, protected * "/")) &&
            throw(ArgumentError("Refusing to write into the archived $name directory; use outputs/runs/<name>."))
    end
end

function acquire_run_lock(out)
    path = joinpath(out, ".run_lock")
    if isdir(path)
        owner_file = joinpath(path, "owner.toml")
        isfile(owner_file) || error("Unowned run lock $path; ensure no runner is active before removing this directory.")
        owner = TOML.parsefile(owner_file)
        owner["host"] == gethostname() || error("Run is locked by another host: $path")
        pid = owner["pid"]
        alive = ccall(:kill, Cint, (Cint, Cint), pid, 0) == 0 || Base.Libc.errno() != 3
        alive && error("Output directory is in use by process $pid")
        rm(path; recursive = true)
    end
    mkdir(path)  # Atomic, so simultaneous writers cannot both acquire this lock.
    atomic_toml(joinpath(path, "owner.toml"), Dict("pid" => getpid(), "host" => gethostname()))
    return path
end

function prepare_run(cfg)
    validate_config(cfg)
    check_output_path(cfg.out)
    identity, fingerprint = run_identity(cfg)
    mkpath(cfg.out)
    manifest = joinpath(cfg.out, "run_manifest.toml")
    if isfile(manifest)
        TOML.parsefile(manifest)["fingerprint"] == fingerprint ||
            error("Run configuration/source/Julia identity differs from this directory. Use a new --out directory.")
    elseif !isempty(readdir(cfg.out))
        error("Refusing a nonempty output directory without run_manifest.toml. Use a new --out directory.")
    end
    lock = acquire_run_lock(cfg.out)
    try
        if !isfile(manifest)
            atomic_toml(manifest, Dict("fingerprint" => fingerprint, "identity" => identity,
                        "created_utc" => string(now(UTC)), "seed_formula" => "data=master_seed+phase*10^15+scenario*10^12+original_case_index*10^10+p*10^5+rep; split=data+5*10^14; basis=basis_master_seed+900000000+1000*p",
                        "basis_hash_format" => "SHA256 of UTF-8 header GDSBasisFloat64LE-v1 rows=<p> cols=<rank> followed by newline, then Float64 little-endian matrix bytes in column-major order",
                        "rng" => "Julia Random.MersenneTwister", "failures" => "Entire replicate is excluded on failure; failure records are retained; run exits unsuccessfully."))
        end
        root = normpath(joinpath(@__DIR__, ".."))
        for (relative, digest) in identity["source_sha256"]
            snapshot = joinpath(cfg.out, "source_snapshot", relative)
            if isfile(snapshot)
                bytes2hex(sha256(read(snapshot))) == digest || error("Source snapshot checksum mismatch: $snapshot")
            else
                bytes = read(joinpath(root, relative))
                bytes2hex(sha256(bytes)) == digest || error("Source changed during run initialization: $relative")
                atomic_write(io -> write(io, bytes), snapshot)
            end
        end
        session = Dict("started_utc" => string(now(UTC)), "pid" => getpid(), "host" => gethostname(),
                       "julia_threads" => Threads.nthreads(), "worker_limit" => cfg.workers,
                       "active_workers" => min(cfg.workers, Threads.nthreads()), "blas_threads" => BLAS.get_num_threads(),
                       "blas_config" => string(BLAS.get_config()), "cpu" => Sys.CPU_NAME,
                       "cpu_threads" => Sys.CPU_THREADS, "ram_bytes" => Sys.total_memory(),
                       "kernel" => string(Sys.KERNEL), "machine" => Sys.MACHINE,
                       "retry_failures" => cfg.retry_failures)
        atomic_toml(joinpath(cfg.out, "sessions", "$(time_ns())_$(getpid()).toml"), session)
    catch
        rm(lock; recursive = true)
        rethrow()
    end
    return fingerprint, lock
end

function validate_replicate_result(result, cfg)
    rows, alignments = result
    length(rows) == 3length(cfg.epsilons) + 2 || error("Incomplete method outcomes")
    length(alignments) == cfg.D || error("Incomplete alignment outcomes")
    all(r -> isfinite(r.chordal_loss) && isfinite(r.wasserstein_loss) && r.selected_rank >= 0, rows) || error("Nonfinite or invalid numerical result")
    all(r -> !(r.method in ("BEMA0", "GD")) || (isfinite(r.sigma2_estimate) && r.sigma2_estimate >= 0), rows) ||
        error("Nonfinite or negative spectral noise-variance estimate")
    all(r -> isfinite(r.q) && isfinite(r.nuclear_prefix), alignments) || error("Nonfinite alignment result")
    return result
end

function csv_field(value)
    text = string(value)
    return occursin(r"[\",\n\r]", text) ? "\"" * replace(text, "\"" => "\"\"") * "\"" : text
end

function atomic_rows(path, rows; columns = nothing)
    names = isempty(rows) ? columns : keys(first(rows))
    names === nothing && return
    atomic_write(path) do io
        println(io, join(string.(names), ','))
        for row in rows
            println(io, join(csv_field.(values(row)), ','))
        end
    end
end

function run_study(cfg::StudyConfig; replicate_fn = replicate)
    BLAS.set_num_threads(1)
    fingerprint, lock = prepare_run(cfg)
    failures, timings, counts, basis_checks = NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[]
    started = time()
    try
        expected = cfg.reps * length(cfg.gammas) * sum(length(selected_conditions(cfg, s)) for s in cfg.scenarios)
        atomic_toml(joinpath(cfg.out, "run_status.toml"), Dict("fingerprint" => fingerprint,
                    "state" => "running", "expected" => expected, "complete" => false,
                    "started_utc" => string(now(UTC))))
        println("Julia $(VERSION); phase=$(cfg.phase); n=$(cfg.n); repetitions=$(cfg.reps); workers=$(min(cfg.workers, Threads.nthreads())); Julia threads=$(Threads.nthreads()); BLAS=1")
        println("Run identity=$fingerprint; output=$(abspath(cfg.out))")
        flush(stdout)
        for gamma in cfg.gammas
            p = round(Int, cfg.n * gamma)
            constants = spectral_constants(cfg, p)
            all(constants.hcuts .< p) || throw(ArgumentError("H_D threshold must be smaller than p"))
            calibration = [(; n = cfg.n, p, gamma = p / cfg.n, D = cfg.D, epsilon = e,
                            H_quantile = h, eta_theta = 1 - 2h / p, eta_w = 1 - 2sqrt(h / p), TW_quantile = tw)
                           for (e, h, tw) in zip(cfg.epsilons, constants.hcuts, constants.twcuts)]
            atomic_rows(joinpath(cfg.out, "calibration_p$(p).csv"), calibration)
            for scenario in cfg.scenarios, (ci, condition) in selected_conditions(cfg, scenario)
                U = population_basis(cfg, p, length(condition.spikes))
                push!(basis_checks, (; scenario, case = condition.tag, n = cfg.n, p,
                                     true_rank = length(condition.spikes),
                                     basis_master_seed = effective_basis_master(cfg),
                                     basis_seed = population_basis_seed(cfg, p), basis_sha256 = basis_digest(U)))
                atomic_rows(joinpath(cfg.out, "basis_verification.csv"), basis_checks)
                stem = "S$(scenario)_$(condition.tag)_p$(p)"
                checkpoint_dir = joinpath(cfg.out, "checkpoints", stem)
                mkpath(checkpoint_dir)
                outcomes = Vector{Any}(undef, cfg.reps)
                pending = Int[]
                for rep in 1:cfg.reps
                    path = joinpath(checkpoint_dir, "$rep.bin")
                    if isfile(path)
                        outcomes[rep] = checkpoint_read(path, fingerprint, (scenario, ci, p, rep))
                        outcomes[rep].status == "failed" && cfg.retry_failures && push!(pending, rep)
                    else
                        push!(pending, rep)
                    end
                end
                elapsed = @elapsed begin
                    workers = min(cfg.workers, Threads.nthreads(), max(1, length(pending)))
                    @sync for worker in 1:workers
                        Threads.@spawn for index in worker:workers:length(pending)
                            rep = pending[index]
                            key = (scenario, ci, p, rep)
                            data_seed, split_seed, basis_seed = replicate_seeds(cfg, scenario, ci, p, rep)
                            tick = time_ns()
                            outcome = try
                                result = validate_replicate_result(replicate_fn(cfg, scenario, ci, condition, p, rep, U, constants), cfg)
                                (; fingerprint, key, status = "ok", result, seconds = (time_ns() - tick) / 1e9)
                            catch err
                                err isa InterruptException && rethrow()
                                failure = (; phase = cfg.phase, scenario, case = condition.tag, n = cfg.n, p, rep,
                                            data_seed, split_seed, basis_seed, error_type = string(typeof(err)), error_message = sprint(showerror, err))
                                (; fingerprint, key, status = "failed", failure, seconds = (time_ns() - tick) / 1e9)
                            end
                            path = joinpath(checkpoint_dir, "$rep.bin")
                            if isfile(path) # Preserve the earlier failed attempt when retrying.
                                cp(path, path * ".failed.$(time_ns())"; force = false)
                            end
                            checkpoint_write(path, outcome)
                            outcomes[rep] = outcome
                        end
                    end
                end
                successful = [o for o in outcomes if o.status == "ok"]
                rows = isempty(successful) ? NamedTuple[] : reduce(vcat, (o.result[1] for o in successful))
                alignments = isempty(successful) ? NamedTuple[] : reduce(vcat, (o.result[2] for o in successful))
                for suffix in ("replicates", "alignments")
                    path = joinpath(cfg.out, stem * "_$suffix.csv")
                    if isempty(successful)
                        isfile(path) && rm(path)
                    else
                        atomic_rows(path, suffix == "replicates" ? rows : alignments)
                    end
                end
                append!(failures, [o.failure for o in outcomes if o.status == "failed"])
                append!(timings, [(; scenario, case = condition.tag, p, rep = o.key[4], status = o.status, seconds = o.seconds) for o in outcomes])
                push!(counts, (; scenario, case = condition.tag, p, expected = cfg.reps, successful = length(successful), failed = cfg.reps - length(successful), computed_this_session = length(pending), wall_seconds_this_session = elapsed))
                @printf("Completed %-28s successful=%d failed=%d computed=%d time=%.2fs\n", stem, length(successful), cfg.reps - length(successful), length(pending), elapsed)
                flush(stdout)
            end
        end
        atomic_rows(joinpath(cfg.out, "failures.csv"), failures; columns = (:phase, :scenario, :case, :n, :p, :rep, :data_seed, :split_seed, :basis_seed, :error_type, :error_message))
        atomic_rows(joinpath(cfg.out, "timings.csv"), timings)
        atomic_rows(joinpath(cfg.out, "completion.csv"), counts)
        atomic_toml(joinpath(cfg.out, "run_status.toml"), Dict("fingerprint" => fingerprint, "finished_utc" => string(now(UTC)),
                    "state" => isempty(failures) ? "completed" : "failed",
                    "expected" => sum(c.expected for c in counts), "successful" => sum(c.successful for c in counts),
                    "failed" => length(failures), "wall_seconds_this_session" => time() - started,
                    "complete" => isempty(failures)))
        isempty(failures) || error("$(length(failures)) replicate(s) failed. See failures.csv; retry explicitly with --retry-failures true.")
    finally
        rm(lock; recursive = true)
    end
    return nothing
end
