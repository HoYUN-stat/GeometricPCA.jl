@testset "semiortho.jl" begin

    @testset "TallSkinnyIdentity" begin
        p, d = 5, 3
        T = Float64
        ts_identity = GeometricPCA.TallSkinnyIdentity(p, d; T=T)

        @testset "Constructors and Basic Properties" begin
            @test size(ts_identity) == (p, d)
            @test eltype(ts_identity) == T
            @test_throws AssertionError GeometricPCA.TallSkinnyIdentity(d, p) # p must be >= d
        end

        @testset "getindex" begin
            @test ts_identity[1, 1] == one(T)
            @test ts_identity[2, 2] == one(T)
            @test ts_identity[3, 3] == one(T)
            @test ts_identity[1, 2] == zero(T)
            @test ts_identity[4, 1] == zero(T)
            @test_throws BoundsError ts_identity[p+1, 1]
        end

        @testset "mul! with Adjoint" begin
            A = rand(p, d)
            C = zeros(d, d)
            At = adjoint(ts_identity)
            GeometricPCA.mul!(C, At, A)
            @test C == A[1:d, :]
        end
    end

    @testset "orthogonalize" begin
        p, d = 4, 2
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
        V = GeometricPCA.orthogonalize(A)

        @test V isa GeometricPCA.SemiOrthogonal
        @test size(parent(V)) == (p, d)

        # Check for orthogonality
        @test parent(V)' * parent(V) ≈ I(d)

        # Test assertion for p < d
        @test_throws AssertionError GeometricPCA.orthogonalize(randn(d, p))
    end

    @testset "SemiOrthogonal" begin
        p, d = 5, 3

        @testset "Lazy Constructor" begin
            V_lazy = GeometricPCA.SemiOrthogonal(p, d)
            @test size(V_lazy) == (p, d)
            @test parent(V_lazy) isa GeometricPCA.TallSkinnyIdentity
            @test V_lazy[1, 1] == 1.0
            @test V_lazy[1, 2] == 0.0
        end

        @testset "Wrapper Constructor and Basic Methods" begin
            M = randn(p, d)
            V_wrapper = GeometricPCA.SemiOrthogonal(M)
            @test parent(V_wrapper) === M
            @test transpose(V_wrapper) isa Transpose
            @test adjoint(V_wrapper) isa Adjoint
        end
    end

    @testset "schatten_norm_pow" begin
        p, d1, d2 = 10, 3, 4
        U = GeometricPCA.orthogonalize(randn(p, d1))
        V = GeometricPCA.orthogonalize(randn(p, d2))
        C = parent(U)' * parent(V)

        @testset "p=2 (Squared Frobenius)" begin
            norm_pow_2 = GeometricPCA.schatten_norm_pow(U, V; p=2)
            @test norm_pow_2 ≈ sum(abs2, C)
        end

        @testset "p=1 (Nuclear Norm)" begin
            norm_pow_1 = GeometricPCA.schatten_norm_pow(U, V; p=1)
            @test norm_pow_1 ≈ sum(svdvals(C))
        end

        @testset "Assertions" begin
            @test_throws AssertionError GeometricPCA.schatten_norm_pow(U, V; p=3)
            W = GeometricPCA.orthogonalize(randn(p + 1, d2))
            @test_throws AssertionError GeometricPCA.schatten_norm_pow(U, W)
        end
    end

    @testset "geometric_risk" begin
        p, D, k = 10, 3, 3
        U = GeometricPCA.SemiOrthogonal(p, D)
        V_matrix = [I(D); zeros(p - D, D)] # Aligned with U
        V = GeometricPCA.SemiOrthogonal(V_matrix)

        @testset "Perfect Alignment" begin
            @test GeometricPCA.geometric_risk(U, V; p=2) ≈ 0.0
            @test GeometricPCA.geometric_risk(U, V; p=1) ≈ 0.0
        end

        @testset "Total Misalignment" begin
            V_misaligned_matrix = [zeros(D, D); I(D); zeros(p - 2D, D)]
            V_misaligned = GeometricPCA.SemiOrthogonal(V_misaligned_matrix)
            @test GeometricPCA.geometric_risk(U, V_misaligned; p=2) ≈ 2D
        end

        @testset "Scaling" begin
            @test GeometricPCA.geometric_risk(U, V; p=2, scaled=true) ≈ 0.0
            @test GeometricPCA.geometric_risk(GeometricPCA.SemiOrthogonal(p, D), V; p=2, scaled=true) ≈ 0.0
        end
    end

    @testset "geometric_risk_sequential" begin
        p, D, k_V = 10, 4, 6
        U = GeometricPCA.SemiOrthogonal(p, D)
        V = GeometricPCA.orthogonalize(randn(p, k_V))

        @testset "Output Properties" begin
            risk_curve_p2 = GeometricPCA.geometric_risk_sequential(U, V; p=2)
            @test length(risk_curve_p2) == k_V

            risk_curve_p1 = GeometricPCA.geometric_risk_sequential(U, V; p=1)
            @test length(risk_curve_p1) == k_V
        end

        @testset "Correctness Check" begin
            # The last value of the sequential risk should equal the direct calculation
            risk_curve = GeometricPCA.geometric_risk_sequential(U, V; p=2, scaled=false)
            direct_risk = GeometricPCA.geometric_risk(U, V; p=2, scaled=false)
            @test risk_curve[end] ≈ direct_risk
        end

        @testset "Lazy U version" begin
            risk_curve = GeometricPCA.geometric_risk_sequential(p, D, V; p=2, scaled=false)
            direct_risk = GeometricPCA.geometric_risk(U, V; p=2, scaled=false)
            @test risk_curve[end] ≈ direct_risk
        end
    end
end
