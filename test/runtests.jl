using GeometricPCA
using Test

@testset "GeometricPCA.jl" begin
    @testset "generate_spiked_data" begin
        n, p, d = 100, 50, 5
        X = generate_spiked_data(n, p, d, 10.0; noise_var=1.0)
        @test size(X) == (n, p)
    end
end
