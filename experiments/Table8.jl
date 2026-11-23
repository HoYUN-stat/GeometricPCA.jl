# Run from the repository root with:
#   julia --project=experiments experiments/compare_lung_cancer_methods.jl

include(joinpath(@__DIR__, "analyze_lung_cancer_gd.jl"))
include(joinpath(@__DIR__, "analyze_lung_cancer_ddpa.jl"))
include(joinpath(@__DIR__, "analyze_lung_cancer_bema.jl"))

const COMPARISON_FILE = joinpath(@__DIR__, "datasets", "lung_cancer_method_comparison.csv")
const LATEX_TABLE_FILE = joinpath(@__DIR__, "tables", "lung_cancer_method_comparison.tex")

function write_latex_table(path, rows)
    mkpath(dirname(path))
    row_end = string(Char(92), Char(92))
    open(path, "w") do io
        println(io, raw"\begin{table}[htbp]")
        println(io, raw"\centering")
        println(io, raw"\small")
        println(io, raw"\caption{Selected dimensions for the lung cancer datasets. GPCA-C and GPCA-BW use the chordal and Bures--Wasserstein split-sample risks, respectively.}")
        println(io, raw"\label{tab:lung-cancer-method-comparison}")
        println(io, raw"\begin{tabular}{lrrrrr}")
        println(io, raw"\hline")
        println(io, "Dataset & GPCA-C & GPCA-BW & GD & DDPA & BEMA $row_end")
        println(io, raw"\hline")
        for row in rows
            dataset = replace(row.dataset, "_" => raw"\_")
            println(io, "$dataset & $(row.geometricpca_chordal) & $(row.geometricpca_bw) & $(row.gavish_donoho) & $(row.ddpa) & $(row.bema) $row_end")
        end
        println(io, raw"\hline")
        println(io, raw"\end{tabular}")
        println(io, raw"\end{table}")
    end
    return path
end

function main()
    gd_rows = run_gd_analysis()
    ddpa_rows = run_ddpa_analysis()
    bema_rows = run_bema_analysis()
    gd_by_dataset = Dict(row.dataset => row for row in gd_rows)
    ddpa_by_dataset = Dict(row.dataset => row for row in ddpa_rows)
    bema_by_dataset = Dict(row.dataset => row for row in bema_rows)

    comparison = NamedTuple[]
    for dataset in LUNG_DATASETS
        X = centered_lung_matrix(dataset.path)
        gpca = geometricpca_ranks(X)
        gd = gd_by_dataset[dataset.name]
        ddpa = ddpa_by_dataset[dataset.name]
        bema = bema_by_dataset[dataset.name]
        push!(comparison, (; dataset = dataset.name,
            geometricpca_chordal = gpca.chordal,
            geometricpca_bw = gpca.bw,
            gavish_donoho = gd.selected_dimension,
            ddpa = ddpa.selected_dimension,
            bema = bema.selected_dimension))
        println("$(dataset.name): GeometricPCA chordal=$(gpca.chordal), BW=$(gpca.bw)")
    end

    write_result_rows(COMPARISON_FILE, comparison)
    println("Saved combined comparison table to $COMPARISON_FILE")
    write_latex_table(LATEX_TABLE_FILE, comparison)
    println("Saved LaTeX table to $LATEX_TABLE_FILE")
end

main()
