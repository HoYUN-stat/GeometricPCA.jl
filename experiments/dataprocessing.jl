# Download and preprocess the complete 1000 Genomes Phase 3 source for the
# genome analysis described in the BEMA paper.
#
# Run from the repository root:
#   julia --project=experiments experiments/dataprocessing.jl
#
# Requires PLINK 2 and PLINK 1.9 on PATH (or set PLINK2/PLINK1 to executable
# paths). PLINK 2 imports the Phase 3 source; PLINK 1.9 runs the reported
# --indep-pairwise pruning procedure. The original
# .bed/.bim/.fam common_norel files are intentionally not used: that derivative
# removed 14 related samples and retained only a subset of Phase 3 variants.

using JLD2
using Statistics

const DATA_DIR = joinpath(@__DIR__, "1000G_data")
const SOURCE_DIR = joinpath(DATA_DIR, "phase3_full")
const DATASET_DIR = joinpath(@__DIR__, "datasets")
const OUTPUT_PATH = joinpath(DATASET_DIR, "1000G_matrix_BEMA_preprocessing.jld2")

const SOURCE_FILES = (
    ("all_phase3.pgen.zst", "https://www.dropbox.com/s/y6ytfoybz48dc0u/all_phase3.pgen.zst?dl=1"),
    ("all_phase3.pvar.zst", "https://www.dropbox.com/s/c95n8quqwqww4s0/all_phase3_noannot.pvar.zst?dl=1"),
    ("all_phase3.psam", "https://www.dropbox.com/scl/fi/haqvrumpuzfutklstazwk/phase3_corrected.psam?dl=1&rlkey=0yyifzj2fb863ddbmsv4jkeq6"),
)

const TARGET_N = 2504
const VARIANT_ID_TEMPLATE = "@:#:\$r:\$a"

function require_plink2()
    path = get(ENV, "PLINK2", "")
    if isempty(path)
        path = something(Sys.which("plink2"), joinpath(DATA_DIR, "tools", "plink2"))
    end
    isempty(path) && error("PLINK 2 was not found on PATH. Install PLINK 2 or set PLINK2 to its executable path.")
    isfile(path) || Sys.which(path) !== nothing || error("PLINK2 does not point to an executable: $path")
    return path
end

function require_plink1()
    path = get(ENV, "PLINK1", "")
    if isempty(path)
        path = something(Sys.which("plink"), joinpath(DATA_DIR, "tools", "plink"))
    end
    isempty(path) && error("PLINK 1.9 was not found on PATH. Install PLINK 1.9 or set PLINK1 to its executable path.")
    isfile(path) || Sys.which(path) !== nothing || error("PLINK1 does not point to an executable: $path")
    return path
end

function download_source()
    mkpath(SOURCE_DIR)
    for (filename, url) in SOURCE_FILES
        destination = joinpath(SOURCE_DIR, filename)
        isfile(destination) && filesize(destination) > 0 && continue

        partial = destination * ".download"
        println("Downloading $filename")
        run(`curl --fail --location --retry 3 --continue-at - --output $partial $url`)
        filesize(partial) > 0 || error("Downloaded file is empty: $partial")
        mv(partial, destination; force=true)
    end
end

function decompress(plink2, compressed_path, output_path)
    run(`$plink2 --zst-decompress $compressed_path $output_path`)
end

function sample_count(psam_path)
    return count(line -> !isempty(strip(line)) && !startswith(strip(line), "#"), eachline(psam_path))
end

function read_plink_raw(path, n_samples)
    open(path, "r") do io
        header = split(strip(readline(io)))
        metadata_columns = 6 # FID, IID, paternal ID, maternal ID, sex, phenotype
        p = length(header) - metadata_columns
        p > 0 || error("No SNP columns found in PLINK raw file: $path")

        X = Matrix{Float64}(undef, n_samples, p)
        for i in 1:n_samples
            eof(io) && error("PLINK raw file ended after $(i - 1) rows; expected $n_samples")
            fields = split(strip(readline(io)))
            length(fields) == length(header) || error("Unexpected column count on PLINK raw row $i")
            for j in 1:p
                value = tryparse(Float64, fields[metadata_columns + j])
                X[i, j] = value === nothing ? NaN : value
            end
        end
        !eof(io) && !isempty(strip(readline(io))) && error("PLINK raw file contains more than $n_samples samples")

        # The Phase 3 files can contain missing calls. Mean-impute within each
        # SNP, as in the previous processing path, while retaining 0/1/2 dosage scale.
        for j in axes(X, 2)
            observed = filter(isfinite, view(X, :, j))
            isempty(observed) && error("SNP column $j has no observed genotype calls")
            replacement = mean(observed)
            for i in axes(X, 1)
                isfinite(X[i, j]) || (X[i, j] = replacement)
            end
        end
        return X
    end
end

function main()
    plink2 = require_plink2()
    plink1 = require_plink1()
    download_source()
    n_samples = sample_count(joinpath(SOURCE_DIR, "all_phase3.psam"))
    n_samples == TARGET_N || error(
        "Phase 3 source contains $n_samples samples; BEMA reports $TARGET_N. " *
        "Check that phase3_corrected.psam downloaded completely and is the expected source.")

    # PLINK 2 consumes the decompressed PGEN/PVAR alongside the supplied PSAM.
    # Keep the large uncompressed intermediates in a temporary directory.
    scratch = mktempdir()
    pgen_prefix = joinpath(scratch, "all_phase3")
    decompress(plink2, joinpath(SOURCE_DIR, "all_phase3.pgen.zst"), pgen_prefix * ".pgen")
    decompress(plink2, joinpath(SOURCE_DIR, "all_phase3.pvar.zst"), pgen_prefix * ".pvar")
    cp(joinpath(SOURCE_DIR, "all_phase3.psam"), pgen_prefix * ".psam")

    filtered_prefix = joinpath(scratch, "1000G_bema_maf_filtered")
    prune_prefix = joinpath(scratch, "1000G_bema")
    prune_file = prune_prefix * ".prune.in"
    raw_prefix = joinpath(scratch, "1000G_bema_pruned")
    raw_file = raw_prefix * ".raw"

    # Keep common autosomal biallelic SNPs, then use PLINK 1.9's independent
    # pairwise pruning: 1000-variant window, step 50, r^2 threshold 0.02.
    run(`$plink2 --pfile $pgen_prefix --set-all-var-ids $VARIANT_ID_TEMPLATE --rm-dup exclude-all --maf 0.01 --autosome --snps-only just-acgt --max-alleles 2 --make-bed --out $filtered_prefix`)
    run(`$plink1 --bfile $filtered_prefix --indep-pairwise 1000 50 0.02 --out $prune_prefix`)
    isfile(prune_file) || error("PLINK 1.9 did not produce the expected prune list: $prune_file")
    run(`$plink1 --bfile $filtered_prefix --extract $prune_file --recode A --out $raw_prefix`)
    isfile(raw_file) || error("PLINK 1.9 did not produce the expected genotype matrix: $raw_file")

    X = read_plink_raw(raw_file, n_samples)
    size(X, 1) == TARGET_N || error("Expected $TARGET_N samples, got $(size(X, 1))")
    n_features = size(X, 2)
    n_features == 24248 || @warn "BEMA reports 24,248 retained SNPs; this run produced $n_features using the published MAF and LD-pruning parameters. The paper/notebook does not document enough additional variant filtering to explain the difference, so no random SNP downsampling was applied."

    # Keep genotype dosage columns on the natural 0/1/2 scale. BEMA does not
    # z-score SNPs; the covariance analysis uses centered data.
    mkpath(DATASET_DIR)
    jldsave(OUTPUT_PATH; X=X, source="1000 Genomes Phase 3",
        maf_threshold=0.01, ld_window_variants=1000, ld_step=50, ld_r2=0.02,
        pruning_software="PLINK 1.9",
        samples=TARGET_N, features=n_features)
    println("Saved BEMA-preprocessed matrix X (samples × SNPs) with size $(size(X)) to $OUTPUT_PATH")
end

main()
