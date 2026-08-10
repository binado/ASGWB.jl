using Test
using Turing
using Turing.DynamicPPL: VarInfo, getsym
using FlexiChains
using AstroSGWB
using Distributions: logpdf
using AstroSGWBInference: build_turing_model, forward_model,
                          AnalyticInclination, CatalogInclination

_varinfo_symbols(vi) = Set(getsym(vn) for vn in keys(vi))

_log_prior(prior, Λ) = sum(logpdf(prior[k], Λ[k]) for k in keys(prior))

@testset "Turing model smoke test with local adapter" begin
    problem = local_problem_context()
    model = build_turing_model(
        problem.model,
        problem.polarization_power,
        problem.samples,
        problem.fiducials,
        problem.frequencies,
        problem.effective_psd,
        problem.observation_time,
        problem.prior;
        track = false
    )
    forward = forward_model(
        problem.model, problem.polarization_power, problem.samples, problem.fiducials)
    rate, log_weights = problem.model(problem.fiducials, problem.samples)
    @test forward.rate == rate
    @test forward.weights ≈ exp.(log_weights)
    @test forward.spectral_density ≈
          spectral_density(problem.polarization_power, rate; weights = exp.(log_weights))

    tracked = build_turing_model(
        problem.model,
        problem.polarization_power,
        problem.samples,
        problem.fiducials,
        problem.frequencies,
        problem.effective_psd,
        problem.observation_time,
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

    # The fixture's unit PSD carries σ ≈ 9e-5 against a fiducial Sₕ ~ 5e-8, so its
    # residual term is ~1e-7 and the joint is numerically all prior plus
    # normalization -- every likelihood-sensitive assertion below would pass
    # vacuously. Score against σ at the signal scale instead: `build_turing_model`
    # derives σ = effective_psd / √(2 T Δf), so pick the PSD that yields σ = 1e-8.
    nfreq = length(problem.frequencies)
    σ_target = 1.0e-8
    eff_psd = fill(
        σ_target * sqrt(2 * year_to_second(problem.observation_time) *
             frequency_bin_width(problem.frequencies)),
        nfreq)
    σ = fill(σ_target, nfreq)

    _build(;
        kwargs...) = build_turing_model(
        problem.model,
        problem.polarization_power,
        problem.samples,
        problem.fiducials,
        problem.frequencies,
        eff_psd,
        problem.observation_time,
        problem.prior;
        kwargs...
    )

    # `observed` is synthesized at the fiducials, so at the fiducials the
    # residual vanishes and the joint collapses to prior + Gaussian
    # normalization -- but only if the synthesized data and the model that
    # scores it used the same averaging mode.
    zero_residual_logjoint = _log_prior(problem.prior, problem.fiducials) -
                             0.5 * sum(log.(2π .* σ .^ 2))

    @testset "both paths agree for every mode" begin
        for mode in (AnalyticInclination(), CatalogInclination())
            m = _build(; average_mode = mode)
            @test Turing.logjoint(m, problem.fiducials) ≈ zero_residual_logjoint

            # S4: there is no second likelihood implementation to compare against, so
            # score `forward_model`'s spectrum with an inline Gaussian instead. Three
            # lines in the test beats a parallel production code path that can drift.
            Sh = forward_model(
                problem.model, problem.polarization_power, problem.samples, problem.theta;
                average_mode = mode).spectral_density
            observed = forward_model(
                problem.model, problem.polarization_power, problem.samples, problem.fiducials;
                average_mode = mode).spectral_density
            residual = observed .- Sh
            expected = _log_prior(problem.prior, problem.theta) -
                       0.5 * sum((residual ./ σ) .^ 2 .+ log.(2π .* σ .^ 2))
            @test Turing.logjoint(m, problem.theta) ≈ expected rtol = 1.0e-6
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
            observed = forward_model(
                problem.model, problem.polarization_power, problem.samples, problem.fiducials;
                average_mode = CatalogInclination()).spectral_density
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

@testset "constants replace conditioning" begin
    problem = local_problem_context()
    full = build_turing_model(
        problem.model,
        problem.polarization_power,
        problem.samples,
        problem.fiducials,
        problem.frequencies,
        problem.effective_psd,
        problem.observation_time,
        problem.prior
    )
    @test _varinfo_symbols(VarInfo(full)) == Set(keys(problem.prior))

    # Restrict the prior to one parameter and hold the rest fixed at the fiducial. The
    # chain then contains exactly the sampled variable *by construction* -- no DynamicPPL
    # conditioning, no complement computation, no subset validation.
    restricted_prior = (; rate_scale = problem.prior.rate_scale)
    constants = Base.structdiff(problem.fiducials, restricted_prior)
    @test keys(constants) == (:weight_shift,)

    restricted = build_turing_model(
        problem.model,
        problem.polarization_power,
        problem.samples,
        problem.fiducials,
        problem.frequencies,
        problem.effective_psd,
        problem.observation_time,
        restricted_prior;
        constants = constants
    )
    @test _varinfo_symbols(VarInfo(restricted)) == Set((:rate_scale,))

    # At a point whose fixed coordinate equals its constant, the restricted model scores
    # the same likelihood as the full one -- the two differ only by the prior term the
    # fixed variable no longer contributes.
    θ = merge(problem.theta, (; weight_shift = problem.fiducials.weight_shift))
    @test Turing.logjoint(restricted, (; rate_scale = θ.rate_scale)) ≈
          Turing.logjoint(full, θ) -
          logpdf(problem.prior.weight_shift, problem.fiducials.weight_shift)

    # `merge(constants, Λ_sampled)` lets the sampled value win, so an overlapping key
    # would silently shadow the constant the caller asked for. Reject it up front.
    @test_throws ArgumentError build_turing_model(
        problem.model,
        problem.polarization_power,
        problem.samples,
        problem.fiducials,
        problem.frequencies,
        problem.effective_psd,
        problem.observation_time,
        restricted_prior;
        constants = problem.fiducials
    )
end
