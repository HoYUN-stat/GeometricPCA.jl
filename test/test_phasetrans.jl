@testset "phasetrans.jl" begin

    @testset "weight_bbp" begin
        γ = 0.5

        @testset "α = 1.0 (Full Sample)" begin
            s = 1.5
            expected = (1 - 1 / s^2) / (1 + sqrt(γ) / s)
            @test GeometricPCA.weight_bbp(s, γ; α=1.0) ≈ expected
            @test_throws AssertionError GeometricPCA.weight_bbp(0.9, γ; α=1.0) # s must be > 1
        end

        @testset "0 < α < 1 (Data Splitting)" begin
            α = 0.4
            s = 2.0
            s_min = 1 / sqrt(min(α, 1 - α))
            @test s > s_min

            term1 = (1.0 - 1.0 / (α * s^2)) / (1.0 + sqrt(γ) / (α * s))
            term2 = (1.0 - 1.0 / ((1.0 - α) * s^2)) / (1.0 + sqrt(γ) / ((1.0 - α) * s))
            expected = term1 * term2
            @test GeometricPCA.weight_bbp(s, γ; α=α) ≈ expected
            @test_throws AssertionError GeometricPCA.weight_bbp(s_min, γ; α=α) # s must be > s_min
        end

        @test_throws AssertionError GeometricPCA.weight_bbp(2.0, -0.1) # γ > 0
        @test_throws AssertionError GeometricPCA.weight_bbp(2.0, 0.5; α=1.1) # α in (0, 1]
    end

    @testset "upper_idx" begin
        lambdas = [10.0, 5.0, 2.0, 1.1]

        @testset "α = 1.0" begin
            γ = 0.5
            λ_thresh = 1.0 + sqrt(γ) # ~1.707
            @test GeometricPCA.upper_idx(lambdas, γ; α=1.0) == 3
        end

        @testset "α = 0.4" begin
            γ = 0.5
            α = 0.4
            γ_max = γ / min(α, 1 - α) # 0.5 / 0.4 = 1.25
            λ_thresh = 1.0 + sqrt(γ_max) # ~2.118
            @test GeometricPCA.upper_idx(lambdas, γ; α=α) == 2
        end
    end

    @testset "signal_thres" begin
        γ = 0.5

        @testset "α = 1.0" begin
            s_star = (sqrt(γ) + sqrt(γ + 8.0)) / 2.0
            @test GeometricPCA.signal_thres(γ; α=1.0) ≈ s_star
            @test GeometricPCA.weight_bbp(s_star, γ; α=1.0) ≈ 0.5
        end

        @testset "α = 0.5" begin
            s_star = (sqrt(2.0) + 1.0) * (sqrt(γ) + sqrt(γ + 4.0 - 2.0 * sqrt(2.0)))
            @test GeometricPCA.signal_thres(γ; α=0.5) ≈ s_star
        end

        @testset "Bisection for α = 0.4" begin
            α = 0.4
            s_star = GeometricPCA.signal_thres(γ; α=α)
            # The test is that at s_star, the weight should be 0.5
            @test GeometricPCA.weight_bbp(s_star, γ; α=α) ≈ 0.5 atol = 1e-9
        end
    end

    @testset "limit_geometric_risk" begin
        lambdas = [10.0, 5.0, 2.0]
        γ = 0.1
        D, k = 3, 3

        @testset "α=1.0, p=2" begin
            d_plus = GeometricPCA.upper_idx(lambdas, γ; α=1.0) # all 3 survive
            s = (lambdas .- 1.0) ./ sqrt(γ)
            alignments = [GeometricPCA.weight_bbp(s_i, γ; α=1.0) for s_i in s]
            expected_risk = D + k - 2 * sum(alignments)
            @test GeometricPCA.limit_geometric_risk(lambdas, γ, D, k; p=2, α=1.0) ≈ expected_risk
        end

        @testset "α=0.4, p=1" begin
            α = 0.4
            D, k = 2, 2
            d_plus = GeometricPCA.upper_idx(lambdas, γ; α=α) # d_plus should be 2
            s = (lambdas[1:2] .- 1.0) ./ sqrt(γ)
            alignments = [sqrt(GeometricPCA.weight_bbp(s_i, γ; α=α)) for s_i in s]
            expected_risk = D + k - 2 * sum(alignments)
            @test GeometricPCA.limit_geometric_risk(lambdas, γ, D, k; p=1, α=α) ≈ expected_risk
        end
    end

    @testset "lower_idx" begin
        lambdas = [100.0, 10.0, 5.0, 2.0]
        γ = 0.2

        @testset "α = 1.0" begin
            s_star = GeometricPCA.signal_thres(γ; α=1.0)
            λ_star = 1.0 + s_star * sqrt(γ)
            # Find last index l where lambdas[l] > λ_star
            expected_d_star = findlast(l -> l > λ_star, lambdas)
            @test GeometricPCA.lower_idx(lambdas, γ; α=1.0) == expected_d_star
        end

        @testset "α = 0.5" begin
            α = 0.5
            s_star = GeometricPCA.signal_thres(γ; α=α)
            λ_star = 1.0 + s_star * sqrt(γ)
            expected_d_star = findlast(l -> l > λ_star, lambdas)
            @test GeometricPCA.lower_idx(lambdas, γ; α=α) == expected_d_star
        end

        @testset "No signal strong enough" begin
            lambdas_weak = [1.1, 1.05]
            @test GeometricPCA.lower_idx(lambdas_weak, γ) == 0
        end
    end
end
