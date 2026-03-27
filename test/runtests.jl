using GeometricPCA
using Test
using Random
using Statistics
using LinearAlgebra

# A clean list of all your test files
const TEST_FILES = [
    "test_spike.jl",
    "test_semiortho.jl",
    "test_phasetrans.jl"
]

@testset "GeometricPCA.jl" begin
    for file in TEST_FILES
        include(file)
    end
end