@testset "spike.jl" begin
    n, p = 10_000, 5
    lambdas = [10.0, 5.0]
    d = length(lambdas)
    noise_var = 1.0

    @testset "generate_spiked_data! (in-place)" begin
        rng = Xoshiro(42)
        X = Matrix{Float64}(undef, n, p)

        # Verify the function modifies and returns the pre-allocated matrix
        @test generate_spiked_data!(X, lambdas; noise_var=noise_var, rng=rng) === X

        # Assertions
        @test_throws AssertionError generate_spiked_data!(X, [10.0, 5.0, 4.0, 3.0, 2.0, 1.0]) # d > p
        @test_throws AssertionError generate_spiked_data!(X, [0.5, 0.5]; noise_var=1.0) # lambda <= noise_var
    end

    @testset "generate_spiked_data (out-of-place with explicit lambdas)" begin
        rng = Xoshiro(42)
        X1 = generate_spiked_data(n, p, lambdas; noise_var=noise_var, rng=rng)

        @test size(X1) == (n, p)
    end

    @testset "generate_spiked_data (out-of-place with constant lambda_value)" begin
        rng = Xoshiro(123)
        lambda_val = 15.0

        # Creates lambdas vector of [lambda_val, ...]
        X2 = generate_spiked_data(n, p, d, lambda_val; noise_var=noise_var, rng=rng)

        @test size(X2) == (n, p)
    end
end
