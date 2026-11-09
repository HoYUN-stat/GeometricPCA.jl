"""
Preprocess the Gordon lung cancer expression data using the feature screening
described in Jin and Wang (2016), Influential Features PCA for High Dimensional
Clustering.

The four input files store genes in rows and samples in columns. This script
combines their 181 samples, standardizes each gene across all samples, ranks
genes by the KS score against a standard normal distribution, and retains the
251 highest-ranked genes. Jin and Wang report that their Higher Criticism
threshold selects 251 genes for this dataset; this script uses that reported
selection count rather than simulating the null distribution used to derive
the threshold.

Run from the repository root or any directory with:
    julia --project=experiments experiments/preprocess_lung_jinwang.jl
"""

using Distributions
using Statistics

const DATA_DIR = joinpath(@__DIR__, "datasets")
const OUTPUT_PATH = joinpath(DATA_DIR, "Lung_JinWang2016_p251_n181.csv")
const TARGET_FEATURES = 251

const INPUT_PARTS = [
    ("ADCA", "train", "LungADCA_train_matrix.txt"),
    ("ADCA", "test",  "LungADCA_test_matrix.txt"),
    ("MES",  "train", "LungMES_train_matrix.txt"),
    ("MES",  "test",  "LungMES_test_matrix.txt"),
]

"Read one genes-by-samples whitespace-delimited matrix into a Float64 matrix."
function read_expression_file(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("Input file is empty: $path")

    first_row = split(strip(lines[1]))
    n_features = length(lines)
    n_samples = length(first_row)
    matrix = Matrix{Float64}(undef, n_features, n_samples)

    for (feature, line) in enumerate(lines)
        values = split(strip(line))
        length(values) == n_samples || error(
            "Inconsistent sample count in $path at feature row $feature",
        )
        for sample in 1:n_samples
            matrix[feature, sample] = parse(Float64, values[sample])
        end
    end
    return matrix
end

function main()
    matrices = Matrix{Float64}[]
    cancer_type = String[]
    split_label = String[]
    n_features = nothing

    for (cancer, split_name, filename) in INPUT_PARTS
        path = joinpath(DATA_DIR, filename)
        isfile(path) || error("Input file not found: $path")
        expression = read_expression_file(path)

        if isnothing(n_features)
            n_features = size(expression, 1)
        elseif size(expression, 1) != n_features
            error("$filename has $(size(expression, 1)) features; expected $n_features")
        end

        push!(matrices, expression)
        append!(cancer_type, fill(cancer, size(expression, 2)))
        append!(split_label, fill(split_name, size(expression, 2)))
    end

    # Convert each genes-by-samples input to samples-by-genes, then concatenate
    # the four sample groups while preserving their cancer type and split labels.
    X = reduce(vcat, permutedims.(matrices))
    n, p = size(X)
    n == 181 || error("Expected 181 datapoints, found $n")
    p == n_features || error("Unexpected feature dimension $p")
    p >= TARGET_FEATURES || error("Need at least $TARGET_FEATURES features, found $p")

    # Jin and Wang's W standardizes each feature using its empirical mean and
    # sample standard deviation across all n samples.
    W = Matrix{Float64}(undef, n, p)
    ks_scores = Vector{Float64}(undef, p)
    normal = Normal()

    for feature in 1:p
        column = view(X, :, feature)
        μ = mean(column)
        σ = std(column)
        σ > 0 || error("Feature $feature has zero sample variance")
        z = (column .- μ) ./ σ
        W[:, feature] .= z

        ordered = sort(z)
        distance = 0.0
        for (rank, value) in enumerate(ordered)
            cdf_value = cdf(normal, value)
            distance = max(distance, rank / n - cdf_value, cdf_value - (rank - 1) / n)
        end
        ks_scores[feature] = sqrt(n) * distance
    end

    # The published HCT procedure selects the 251 largest KS scores on this
    # lung dataset. Use source feature index to break any exact score ties.
    ranked = sort(collect(1:p); by=feature -> (-ks_scores[feature], feature))
    selected = sort(ranked[1:TARGET_FEATURES])

    open(OUTPUT_PATH, "w") do io
        header = vcat(
            ["cancer_type", "split"],
            ["gene_$(lpad(feature, 5, '0'))" for feature in selected],
        )
        println(io, join(header, ','))

        for sample in 1:n
            values = vcat(
                [cancer_type[sample], split_label[sample]],
                [string(W[sample, feature]) for feature in selected],
            )
            println(io, join(values, ','))
        end
    end

    println("Wrote $OUTPUT_PATH (n=$n datapoints, p=$(length(selected)) features)")
end

main()
