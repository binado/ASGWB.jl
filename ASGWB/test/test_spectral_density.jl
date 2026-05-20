using ASGWB: Ωgw
using CBCDistributions: hubble_constant_si
using ForwardDiff
using Statistics
using Test

@testset "Ωgw" begin
    H0_kms = 70.0
    h0_si = hubble_constant_si(H0_kms)
    f = [10.0, 20.0]
    sh = [1.0e-45, 2.0e-45]
    pre = 4 * pi^2 / (3 * h0_si^2)
    expected = @. pre * f^3 * sh
    @test Ωgw(sh, f, H0_kms) ≈ expected
    @test Ωgw(1.0e-45, 10.0, H0_kms) ≈ pre * 10.0^3 * 1.0e-45
end

@testset "spectral_density primitive" begin
    fluxes = Float64[1.0 2.0 3.0; 4.0 5.0 6.0]
    rate = 2.5
    n_samples = size(fluxes, 2)

    @testset "unweighted average over samples" begin
        expected = 0.4 .* rate .* vec(mean(fluxes; dims = 2))
        @test spectral_density(fluxes, rate) ≈ expected
    end

    @testset "weighted contraction without weight normalization" begin
        w = [0.5, 1.0, 2.0]
        expected = 0.4 .* rate .* (fluxes * w) ./ n_samples
        @test spectral_density(fluxes, rate; weights = w) ≈ expected
    end

    @testset "uniform weights equal to ones give the unweighted mean" begin
        w = ones(n_samples)
        @test spectral_density(fluxes, rate; weights = w) ≈ spectral_density(fluxes, rate)
    end

    @testset "length mismatch errors from matrix multiply" begin
        @test_throws DimensionMismatch spectral_density(fluxes, rate; weights = [1.0, 2.0])
    end

    @testset "output length matches n_freq" begin
        @test length(spectral_density(fluxes, rate)) == size(fluxes, 1)
        @test length(spectral_density(fluxes, rate; weights = rand(n_samples))) ==
              size(fluxes, 1)
    end

    @testset "dual weighted contraction matches generic expression" begin
        w = [
            ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(1.0, -0.5), ForwardDiff.Dual(2.0, 0.25)]
        expected = 0.4 .* rate .* ((fluxes * w) ./ n_samples)
        got = spectral_density(fluxes, rate; weights = w)
        @test ForwardDiff.value.(got) ≈ ForwardDiff.value.(expected)
        @test [ForwardDiff.partials(x)[1] for x in got] ≈
              [ForwardDiff.partials(x)[1] for x in expected]
    end

    @testset "dual weighted contraction handles dual rate and multiple lanes" begin
        w = [
            ForwardDiff.Dual{Nothing, Float64, 2}(0.5, ForwardDiff.Partials((1.0, 0.1))),
            ForwardDiff.Dual{Nothing, Float64, 2}(1.0, ForwardDiff.Partials((-0.5, 0.2))),
            ForwardDiff.Dual{Nothing, Float64, 2}(2.0, ForwardDiff.Partials((0.25, -0.3)))
        ]
        rate_dual = ForwardDiff.Dual{Nothing, Float64, 2}(rate, ForwardDiff.Partials((
            0.3, -0.1)))
        expected = 0.4 .* rate_dual .* ((fluxes * w) ./ n_samples)
        got = spectral_density(fluxes, rate_dual; weights = w)
        @test ForwardDiff.value.(got) ≈ ForwardDiff.value.(expected)
        for lane in 1:2
            @test [ForwardDiff.partials(x)[lane] for x in got] ≈
                  [ForwardDiff.partials(x)[lane] for x in expected]
        end
    end
end
