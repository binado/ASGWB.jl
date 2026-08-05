using Test
using Turing
using Turing.DynamicPPL: VarInfo, getsym
using FlexiChains
using AstroSGWB
using Distributions: logpdf
using AstroSGWBInference: build_turing_model, condition_turing_model,
                          fiducial_spectral_density, logposterior,
                          AnalyticInclination, CatalogInclination

_varinfo_symbols(vi) = Set(getsym(vn) for vn in keys(vi))

@testset "Turing model smoke test with local adapter" begin
    problem = local_problem_context()
    model = build_turing_model(
        problem.model,
        problem.fluxes,
        problem.samples,
        problem.fiducials,
        problem.observation,
        problem.prior;
        track = false
    )
    observed = fiducial_spectral_density(
        problem.model, problem.fluxes, problem.samples, problem.fiducials)
    rate, log_weights = problem.model(problem.fiducials, problem.samples)
    @test observed ≈ spectral_density(problem.fluxes, rate; weights = exp.(log_weights))
    @test Turing.logjoint(model, problem.theta) ≈ logposterior(
        problem.theta,
        problem.model,
        problem.fluxes,
        problem.samples,
        problem.observation,
        problem.prior,
        observed
    ) rtol = 1.0e-6

    tracked = build_turing_model(
        problem.model,
        problem.fluxes,
        problem.samples,
        problem.fiducials,
        problem.observation,
        problem.prior;
        track = true
    )
    returned_nt = Turing.returned(tracked, problem.theta)
    @test 0 < returned_nt.effective_sample_size <= 1
    @test isfinite(returned_nt.spectral_snr)
    @test returned_nt.spectral_snr^2 ≈ returned_nt.spectral_snr_squared

    chain = sample(
        model,
        Turing.NUTS(3, 0.8),
        3;
        progress = false,
        chain_type = FlexiChains.VNChain,
        initial_params = InitFromPrior()
    )
    @test chain isa FlexiChains.VNChain
    @test size(chain, 1) == 3
    @test sort(collect(Symbol.(FlexiChains.parameters(chain)))) ==
          sort(collect(keys(problem.theta)))
    @test all(isfinite, vec(Array(chain[:logjoint])))
end

@testset "average_mode reaches both sides of build_turing_model" begin
    problem = local_problem_context()

    # `LOCAL_OBSERVATION` carries σ = 1 against a fiducial Sₕ ~ 5e-8, so its
    # residual term is ~1e-15 and the joint is numerically all prior plus
    # normalization -- every likelihood-sensitive assertion below would pass
    # vacuously. Score against σ at the signal scale instead.
    observation = ObservationContext(
        problem.observation.frequencies,
        problem.observation.effective_psd,
        fill(1.0e-8, length(problem.observation.frequencies)),
        problem.observation.in_band_mask,
        problem.observation.observation_time
    )
    σ = observation.sgwb_scale_in_band

    _build(;
        kwargs...) = build_turing_model(
        problem.model,
        problem.fluxes,
        problem.samples,
        problem.fiducials,
        observation,
        problem.prior;
        kwargs...
    )

    # `observed` is synthesized at the fiducials, so at the fiducials the
    # residual vanishes and the joint collapses to prior + Gaussian
    # normalization -- but only if the synthesized data and the model that
    # scores it used the same averaging mode.
    zero_residual_logjoint = sum(
        logpdf(problem.prior[k], problem.fiducials[k])
    for k in keys(problem.prior)) -
                             0.5 * sum(log.(2π .* σ .^ 2))

    @testset "both paths agree for every mode" begin
        for mode in (AnalyticInclination(), CatalogInclination())
            m = _build(; average_mode = mode)
            @test Turing.logjoint(m, problem.fiducials) ≈ zero_residual_logjoint

            observed = fiducial_spectral_density(
                problem.model, problem.fluxes, problem.samples, problem.fiducials;
                average_mode = mode)
            @test Turing.logjoint(m, problem.theta) ≈ logposterior(
                problem.theta,
                problem.model,
                problem.fluxes,
                problem.samples,
                observation,
                problem.prior,
                observed;
                average_mode = mode
            ) rtol = 1.0e-6
        end
    end

    @testset "the modes are not observationally equivalent" begin
        @test !isapprox(
            Turing.logjoint(_build(; average_mode = AnalyticInclination()), problem.theta),
            Turing.logjoint(_build(; average_mode = CatalogInclination()), problem.theta)
        )
    end

    # Proves the zero-residual assertion above is not vacuous: feed the model an
    # `observed` built under the other convention and the identity must break.
    @testset "a mismatched pair fails the zero-residual identity" begin
        mismatched = _build(;
            average_mode = AnalyticInclination(),
            observed = fiducial_spectral_density(
                problem.model, problem.fluxes, problem.samples, problem.fiducials;
                average_mode = CatalogInclination())
        )
        @test !isapprox(
            Turing.logjoint(mismatched, problem.fiducials), zero_residual_logjoint)
    end

    @testset "the mode is visible in the built model's positional args" begin
        @test _build(; average_mode = CatalogInclination()).args.average_mode ===
              CatalogInclination()
    end

    @testset "the default is AnalyticInclination" begin
        default = _build()
        @test default.args.average_mode === AnalyticInclination()
        @test Turing.logjoint(default, problem.theta) ≈
              Turing.logjoint(
            _build(; average_mode = AnalyticInclination()), problem.theta)
    end
end

@testset "flat submodel and conditioning boundary" begin
    problem = local_problem_context()
    model = build_turing_model(
        problem.model,
        problem.fluxes,
        problem.samples,
        problem.fiducials,
        problem.observation,
        problem.prior
    )
    present = _varinfo_symbols(VarInfo(model))
    @test present == Set((:rate_scale, :weight_shift))

    @test condition_turing_model(
        model, problem.theta, problem.prior, nothing) === model
    conditioned = condition_turing_model(
        model, problem.theta, problem.prior, (:rate_scale,))
    @test _varinfo_symbols(VarInfo(conditioned)) == Set((:rate_scale,))

    @test_throws ArgumentError condition_turing_model(
        model, problem.theta, problem.prior, ())
    @test_throws ArgumentError condition_turing_model(
        model, problem.theta, problem.prior, (:unknown,))
    @test_throws ArgumentError condition_turing_model(
        model, problem.theta, problem.prior, (:rate_scale, :rate_scale))
end
