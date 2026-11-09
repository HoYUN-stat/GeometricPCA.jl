# Downloading and processing 1000 Genomes dataset.
# This is only being run to create the dataset X,
# which is saved to use for future analysis.

using Pkg

Pkg.add(["SnpArrays", "LinearAlgebra", "Random", "Statistics"])

using SnpArrays, LinearAlgebra, Random, Statistics

bed_path = "./1000G_data/1000G_phase3_common_norel.bed"

if !isfile(bed_path)
    error("BED file not found at $bed_path. Please ensure 1000G_phase3_common_norel.zip is extracted into ./1000G_data/")
end

# 1. Memory-map raw binary PLINK dataset
snp_data = SnpArray(bed_path)
n_samples, n_snps = size(snp_data)
println("Raw dataset loaded: n = $n_samples samples, p = $n_snps SNPs")

# 2. Convert to Float64 matrix (defaults to additive dosage and mean imputation)
X_raw = convert(Matrix{Float64}, snp_data, impute=true)

# 3. Filter by Minor Allele Frequency (MAF >= 0.05)
allele_freqs = mean(X_raw, dims=1) ./ 2.0
mafs = min.(allele_freqs, 1.0 .- allele_freqs)

valid_snp_indices = findall(vec(mafs .>= 0.05))
println("SNPs passing MAF >= 0.05 filter: ", length(valid_snp_indices))

# 4. Downsample to target BEMA dimensions (24,249 SNPs)
Random.seed!(42)
selected_snps = Random.shuffle(valid_snp_indices)[1:24249]

X_sub = X_raw[:, selected_snps]

# 5. Standardize genotypes (mean = 0, std = 1 per column)
X_scaled = (X_sub .- mean(X_sub, dims=1)) ./ std(X_sub, dims=1)

n, p = size(X_scaled)
println("Final matrix dimensions (n x p): ($n, $p)")

using Pkg
Pkg.add("JLD2")

using JLD2

# Save the matrix to a binary file
# 'iowrite' or 'compress' can be used to keep disk footprint small (~200–400 MB)
jldsave("1000G_matrix_2504x24249.jld2", compress = true; X = X_scaled)

println("Saved X_scaled to 1000G_matrix_2504x24249.jld2")