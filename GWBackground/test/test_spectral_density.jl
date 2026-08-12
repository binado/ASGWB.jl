using GWBackground: Ωgw, spectral_density, AnalyticInclination, CatalogInclination,
                 inclination_factor, average_mode_config_name, average_mode_type,
                 SUPPORTED_AVERAGE_MODES
using BackgroundCosmology: hubble_constant_si
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
    polarization_power = Float64[1.0 2.0 3.0; 4.0 5.0 6.0]
    rate = 2.5
    nsamples = size(polarization_power, 2)

    @testset "unweighted average over samples" begin
        expected = 0.4 .* rate .* vec(mean(polarization_power; dims = 2))
        @test spectral_density(polarization_power, rate) ≈ expected
    end

    @testset "weighted contraction without weight normalization" begin
        w = [0.5, 1.0, 2.0]
        expected = 0.4 .* rate .* (polarization_power * w) ./ nsamples
        @test spectral_density(polarization_power, rate; weights = w) ≈ expected
    end

    @testset "uniform weights equal to ones give the unweighted mean" begin
        w = ones(nsamples)
        @test spectral_density(polarization_power, rate; weights = w) ≈
              spectral_density(polarization_power, rate)
    end

    @testset "length mismatch errors from matrix multiply" begin
        @test_throws DimensionMismatch spectral_density(polarization_power, rate; weights = [
            1.0, 2.0])
    end

    @testset "output length matches nfreq" begin
        @test length(spectral_density(polarization_power, rate)) ==
              size(polarization_power, 1)
        @test length(spectral_density(polarization_power, rate; weights = rand(nsamples))) ==
              size(polarization_power, 1)
    end

    @testset "dual weighted contraction matches generic expression" begin
        w = [
            ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(1.0, -0.5), ForwardDiff.Dual(2.0, 0.25)]
        expected = 0.4 .* rate .* ((polarization_power * w) ./ nsamples)
        got = spectral_density(polarization_power, rate; weights = w)
        @test ForwardDiff.value.(got) ≈ ForwardDiff.value.(expected)
        @test getindex.(ForwardDiff.partials.(got), 1) ≈
              getindex.(ForwardDiff.partials.(expected), 1)
    end

    @testset "dual weighted contraction handles dual rate and multiple lanes" begin
        w = [
            ForwardDiff.Dual{Nothing, Float64, 2}(0.5, ForwardDiff.Partials((1.0, 0.1))),
            ForwardDiff.Dual{Nothing, Float64, 2}(1.0, ForwardDiff.Partials((-0.5, 0.2))),
            ForwardDiff.Dual{Nothing, Float64, 2}(2.0, ForwardDiff.Partials((0.25, -0.3)))
        ]
        rate_dual = ForwardDiff.Dual{Nothing, Float64, 2}(rate, ForwardDiff.Partials((
            0.3, -0.1)))
        expected = 0.4 .* rate_dual .* ((polarization_power * w) ./ nsamples)
        got = spectral_density(polarization_power, rate_dual; weights = w)
        @test ForwardDiff.value.(got) ≈ ForwardDiff.value.(expected)
        for lane in 1:2
            @test getindex.(ForwardDiff.partials.(got), lane) ≈
                  getindex.(ForwardDiff.partials.(expected), lane)
        end
    end
end

@testset "average mode tokens" begin
    @test inclination_factor(AnalyticInclination()) == 0.4
    @test inclination_factor(CatalogInclination()) == 1.0

    @testset "config names round-trip and match the astrogwb literals" begin
        @test average_mode_config_name(AnalyticInclination) == "analytic_inclination"
        @test average_mode_config_name(CatalogInclination) == "catalog_inclination"
        for M in SUPPORTED_AVERAGE_MODES
            @test average_mode_type(average_mode_config_name(M)) === M
        end
        @test_throws ArgumentError average_mode_type("face_on")
    end
end

@testset "spectral_density average_mode" begin
    polarization_power = Float64[1.0 2.0 3.0; 4.0 5.0 6.0]
    rate = 2.5
    nsamples = size(polarization_power, 2)
    real_weights = [0.5, 1.0, 2.0]
    dual_weights = [
        ForwardDiff.Dual{Nothing, Float64, 2}(0.5, ForwardDiff.Partials((1.0, 0.1))),
        ForwardDiff.Dual{Nothing, Float64, 2}(1.0, ForwardDiff.Partials((-0.5, 0.2))),
        ForwardDiff.Dual{Nothing, Float64, 2}(2.0, ForwardDiff.Partials((0.25, -0.3)))
    ]

    @testset "the default is AnalyticInclination on every dispatch branch" begin
        @test spectral_density(polarization_power, rate) ≈
              spectral_density(polarization_power, rate; average_mode = AnalyticInclination())
        @test spectral_density(polarization_power, rate; weights = real_weights) ≈
              spectral_density(polarization_power, rate; weights = real_weights,
            average_mode = AnalyticInclination())
        got = spectral_density(polarization_power, rate; weights = dual_weights)
        ref = spectral_density(polarization_power, rate; weights = dual_weights,
            average_mode = AnalyticInclination())
        @test ForwardDiff.value.(got) ≈ ForwardDiff.value.(ref)
        for lane in 1:2
            @test [ForwardDiff.partials(x)[lane] for x in got] ≈
                  [ForwardDiff.partials(x)[lane] for x in ref]
        end
    end

    # The three `_spectral_density` branches carry the prefactor independently;
    # a partial edit that misses one would leave a stale 0.4 in that branch only.
    @testset "all three dispatch branches share one prefactor" begin
        ratio = inclination_factor(CatalogInclination()) /
                inclination_factor(AnalyticInclination())

        @testset "unweighted" begin
            analytic = spectral_density(polarization_power, rate;
                average_mode = AnalyticInclination())
            catalog = spectral_density(polarization_power, rate;
                average_mode = CatalogInclination())
            @test catalog ≈ ratio .* analytic
            @test catalog ≈ rate .* vec(mean(polarization_power; dims = 2))
        end

        @testset "real weights" begin
            analytic = spectral_density(polarization_power, rate; weights = real_weights,
                average_mode = AnalyticInclination())
            catalog = spectral_density(polarization_power, rate; weights = real_weights,
                average_mode = CatalogInclination())
            @test catalog ≈ ratio .* analytic
            @test catalog ≈ rate .* (polarization_power * real_weights) ./ nsamples
        end

        @testset "dual weights: values and partials both scale" begin
            analytic = spectral_density(polarization_power, rate; weights = dual_weights,
                average_mode = AnalyticInclination())
            catalog = spectral_density(polarization_power, rate; weights = dual_weights,
                average_mode = CatalogInclination())
            @test ForwardDiff.value.(catalog) ≈ ratio .* ForwardDiff.value.(analytic)
            # Guards the `ntuple(j -> scale * ...)` line: scaling the primal but
            # not the partials would pass a value-only comparison.
            for lane in 1:2
                @test [ForwardDiff.partials(x)[lane] for x in catalog] ≈
                      ratio .* [ForwardDiff.partials(x)[lane] for x in analytic]
            end
            expected = rate .* ((polarization_power * dual_weights) ./ nsamples)
            @test ForwardDiff.value.(catalog) ≈ ForwardDiff.value.(expected)
            for lane in 1:2
                @test [ForwardDiff.partials(x)[lane] for x in catalog] ≈
                      [ForwardDiff.partials(x)[lane] for x in expected]
            end
        end
    end

    @testset "a dual rate scales with the mode too" begin
        rate_dual = ForwardDiff.Dual{Nothing, Float64, 2}(
            rate, ForwardDiff.Partials((0.3, -0.1)))
        analytic = spectral_density(polarization_power, rate_dual; weights = dual_weights,
            average_mode = AnalyticInclination())
        catalog = spectral_density(polarization_power, rate_dual; weights = dual_weights,
            average_mode = CatalogInclination())
        ratio = inclination_factor(CatalogInclination()) /
                inclination_factor(AnalyticInclination())
        @test ForwardDiff.value.(catalog) ≈ ratio .* ForwardDiff.value.(analytic)
        for lane in 1:2
            @test [ForwardDiff.partials(x)[lane] for x in catalog] ≈
                  ratio .* [ForwardDiff.partials(x)[lane] for x in analytic]
        end
    end
end
