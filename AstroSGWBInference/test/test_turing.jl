using Test
using Turing
using Turing.DynamicPPL: VarInfo, getsym
using FlexiChains
using AstroSGWB
using Distributions: logpdf
using AstroSGWBInference: astrosgwb_importance_turing_model, forward_model,
                          AnalyticInclination, CatalogInclination

_varinfo_symbols(vi) = Set(getsym(vn) for vn in keys(vi))

_log_prior(prior, Λ) = sum(logpdf(prior[k], Λ[k]) for k in keys(prior))

# Constructing the model is caller-owned: synthesize `observed` at the fiducials (or pass
# an external spectrum), call the `@model` directly, and pin fixed hyperparameters by
# conditioning (`model | fixed`). This helper only spares the test bodies the repetition.
function _inline_model(problem; track = false,
        average_mode = AnalyticInclination(),
        prior = problem.prior, fixed = NamedTuple(),
        effective_psd = problem.effective_psd,
        observed = forward_model(
            problem.model, problem.polarization_power, problem.samples, problem.fiducials;
            average_mode = average_mode).spectral_density)
    model = astrosgwb_importance_turing_model(
        problem.model, problem.polarization_power, problem.samples, prior, observed,
        problem.frequencies, effective_psd, problem.observation_time, average_mode, track)
    return model | fixed
end

@testset "Turing model smoke test with local adapter" begin
    problem = local_problem_context()
    model = _inline_model(problem; track = false)
    forward = forward_model(
        problem.model, problem.polarization_power, problem.samples, problem.fiducials)
    rate, log_weights = problem.model(problem.fiducials, problem.samples)
    @test forward.rate == rate
    @test forward.weights ≈ exp.(log_weights)
    @test forward.spectral_density ≈
          spectral_density(problem.polarization_power, rate; weights = exp.(log_weights))

    tracked = _inline_model(problem; track = true)
    returned_nt = Turing.returned(tracked, problem.theta)
    @test 0 < returned_nt.effective_sample_size <= 1
    @test isfinite(returned_nt.snr)
    Sh_theta = forward_model(
        problem.model, problem.polarization_power, problem.samples, problem.theta).spectral_density
    @test returned_nt.snr ≈ spectral_snr(
        Sh_theta, problem.effective_psd,
        year_to_second(problem.observation_time),
        frequency_bin_width(problem.frequencies))

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

@testset "one average_mode reaches both the data and the model" begin
    problem = local_problem_context()

    # The fixture's unit PSD carries σ ≈ 9e-5 against a fiducial Sₕ ~ 5e-8, so its
    # residual term is ~1e-7 and the joint is numerically all prior plus
    # normalization -- every likelihood-sensitive assertion below would pass
    # vacuously. Score against σ at the signal scale instead: the model body
    # derives σ = effective_psd / √(2 T Δf), so pick the PSD that yields σ = 1e-8.
    nfreq = length(problem.frequencies)
    σ_target = 1.0e-8
    eff_psd = fill(
        σ_target * sqrt(2 * year_to_second(problem.observation_time) *
             frequency_bin_width(problem.frequencies)),
        nfreq)
    σ = fill(σ_target, nfreq)

    _build(; kwargs...) = _inline_model(problem; effective_psd = eff_psd, kwargs...)

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

            # There is no second likelihood implementation to compare against, so
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
end

@testset "conditioning pins hyperparameters" begin
    problem = local_problem_context()
    full = _inline_model(problem)
    @test _varinfo_symbols(VarInfo(full)) == Set(keys(problem.prior))

    # The prior declares every name; conditioning on the complement fixes
    # `weight_shift` at the fiducial. The chain then contains exactly the sampled
    # variable *by construction* -- no helpers, no subset validation.
    fixed = (; weight_shift = problem.fiducials.weight_shift)
    restricted = _inline_model(problem; fixed = fixed)
    @test _varinfo_symbols(VarInfo(restricted)) == Set((:rate_scale,))

    # Conditioning moves the pinned variable's prior density from the log-prior into the
    # likelihood, so the conditioned model scored at the free coordinates equals the full
    # model scored at the same point -- exactly, not up to a dropped prior term.
    θ = merge(problem.theta, fixed)
    @test Turing.logjoint(restricted, (; rate_scale = θ.rate_scale)) ≈
          Turing.logjoint(full, θ)

    # A pinned value outside its prior support scores -Inf at the first evaluation. Loud
    # by construction; no support validation anywhere in the pipeline.
    out_of_support = _inline_model(problem; fixed = (; weight_shift = 1.0))
    @test Turing.logjoint(out_of_support, (; rate_scale = θ.rate_scale)) == -Inf
end
