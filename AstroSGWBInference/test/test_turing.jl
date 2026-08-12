using Test
using Turing
using Turing.DynamicPPL: VarInfo, getsym
using FlexiChains
using AstroSGWB
using Distributions: logpdf
using AstroSGWBInference: astrosgwb_importance_turing_model,
                          astrosgwb_amplitude_marginalized_turing_model,
                          forward_model, quadrature_grid,
                          AnalyticInclination, CatalogInclination

_varinfo_symbols(vi) = Set(getsym(vn) for vn in keys(vi))

_log_prior(prior, Λ) = sum(logpdf(prior[k], Λ[k]) for k in keys(prior))

# The two `:=` sites every draw of the general model records. They are parameters as far
# as FlexiChains and the netCDF `posterior` group are concerned, so every assertion about
# what the chain carries has to name them.
const DERIVED_PARAMS = (:total_merger_rate, :importance_relative_ess)

# Constructing the model is caller-owned: synthesize `observed` at the fiducials (or pass
# an external spectrum), call the `@model` directly, and pin fixed hyperparameters by
# conditioning (`model | fixed`). This helper only spares the test bodies the repetition.
function _inline_model(problem;
        average_mode = AnalyticInclination(),
        prior = problem.prior, fixed = NamedTuple(),
        prior_model = toy_prior_model(prior),
        effective_psd = problem.effective_psd,
        observed = forward_model(
            problem.model, problem.polarization_power, problem.samples, problem.fiducials;
            average_mode = average_mode).spectral_density)
    model = astrosgwb_importance_turing_model(
        problem.model, problem.polarization_power, problem.samples, prior_model, observed,
        problem.frequencies, effective_psd, problem.observation_time, average_mode)
    return model | fixed
end

# The fixture's unit PSD carries σ ≈ 9e-5 against a fiducial Sₕ ~ 5e-8, so its residual
# term is ~1e-7 and the joint is numerically all prior plus normalization -- every
# likelihood-sensitive assertion would pass vacuously. Score against σ at the signal scale
# instead: the model body derives σ = effective_psd / √(2 T Δf), so pick the PSD that
# yields σ = 1e-8.
function _signal_scale_psd(problem; σ_target = 1.0e-8)
    return fill(
        σ_target * sqrt(2 * year_to_second(problem.observation_time) *
             frequency_bin_width(problem.frequencies)),
        length(problem.frequencies))
end

# Independent of the package's own quadrature, so the equivalence test below checks the
# marginalization rather than restating its implementation.
function _log_trapezoid(log_y, x)
    m = maximum(log_y)
    y = exp.(log_y .- m)
    return m + log(sum(0.5 .* (y[1:(end - 1)] .+ y[2:end]) .* diff(collect(x))))
end

@testset "Turing model smoke test with local adapter" begin
    problem = local_problem_context()
    model = _inline_model(problem)
    forward = forward_model(
        problem.model, problem.polarization_power, problem.samples, problem.fiducials)
    rate, log_weights = problem.model(problem.fiducials, problem.samples)
    @test forward.rate == rate
    @test forward.weights ≈ exp.(log_weights)
    @test forward.spectral_density ≈
          spectral_density(problem.polarization_power, rate; weights = exp.(log_weights))

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
    # The `:=` sites are parameters, so the chain carries the sampled hyperparameters
    # *plus* the derived quantities -- which is the whole point: they were being computed
    # on every step and thrown away before.
    @test sort(collect(Symbol.(FlexiChains.parameters(chain)))) ==
          sort(collect((keys(problem.theta)..., DERIVED_PARAMS...)))
    @test all(isfinite, vec(Array(chain[:logjoint])))
end

@testset "the := sites record the forward model's own quantities" begin
    problem = local_problem_context()
    model = _inline_model(problem)
    forward = forward_model(
        problem.model, problem.polarization_power, problem.samples, problem.theta)

    chain = sample(
        model,
        Turing.NUTS(3, 0.8),
        3;
        progress = false,
        chain_type = FlexiChains.VNChain,
        initial_params = InitFromPrior()
    )
    for name in DERIVED_PARAMS
        @test all(isfinite, vec(Array(chain[FlexiChains.Parameter(@varname($name))])))
    end
    @test all(0 .<
              vec(Array(chain[FlexiChains.Parameter(@varname(importance_relative_ess))])) .<=
              1)

    # Scored at a specific θ, `total_merger_rate` is exactly the forward model's rate --
    # not a rescaled or time-integrated version of it.
    recorded = Turing.DynamicPPL.ParamsWithStats(
        Turing.DynamicPPL.InitFromParams(problem.theta), model).params
    @test recorded[@varname(total_merger_rate)] == forward.rate
    @test recorded[@varname(importance_relative_ess)] ≈
          inv(sum(abs2, forward.weights ./ sum(forward.weights))) / length(forward.weights)
end

@testset "one average_mode reaches both the data and the model" begin
    problem = local_problem_context()

    nfreq = length(problem.frequencies)
    σ_target = 1.0e-8
    eff_psd = _signal_scale_psd(problem; σ_target)
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

# --------------------------------------------------------------------------
# Amplitude marginalization
# --------------------------------------------------------------------------

# The local adapter's rate is `1e-7 * Λ.rate_scale` and its log-weights do not read
# `rate_scale` at all, so `rate_scale` is exactly multiplicative with f(φ) = φ -- the same
# structure `R₀` has in the production BNS adapter, which
# `AstroSGWBImportanceModels/test` asserts against the real forward model.
const AMPLITUDE_NAME = :rate_scale
const AMPLITUDE_DERIVED = (:template_merger_rate, :amplitude_mle, :template_optimal_snr,
    :importance_relative_ess)

function _marginalized_model(problem;
        average_mode = AnalyticInclination(),
        prior = (; weight_shift = problem.prior.weight_shift),
        prior_model = toy_prior_model_shape_only(prior),
        effective_psd = _signal_scale_psd(problem),
        amplitude_prior = problem.prior.rate_scale,
        amplitude_fiducial = (; rate_scale = problem.fiducials.rate_scale),
        amplitude_fn = identity,
        grid = quadrature_grid(amplitude_prior; num_nodes = 4096),
        observed = forward_model(
            problem.model, problem.polarization_power, problem.samples, problem.fiducials;
            average_mode = average_mode).spectral_density)
    return astrosgwb_amplitude_marginalized_turing_model(
        problem.model, problem.polarization_power, problem.samples, prior_model, observed,
        problem.frequencies, effective_psd, problem.observation_time, average_mode,
        amplitude_fiducial, amplitude_fn, amplitude_prior, grid)
end

@testset "marginalizing == numerically integrating the general model" begin
    # The load-bearing test. If the completed square, the σ-space inner products, the
    # `data_norm` term, or the normalizer disagree with the general likelihood by so much
    # as an additive constant, this fails -- and a constant offset in a log density is
    # exactly the bug that leaves the posterior looking perfectly reasonable.
    problem = local_problem_context()
    eff_psd = _signal_scale_psd(problem)
    amplitude_prior = problem.prior.rate_scale
    grid = quadrature_grid(amplitude_prior; num_nodes = 4096)
    observed = forward_model(
        problem.model, problem.polarization_power, problem.samples,
        problem.fiducials).spectral_density

    general = astrosgwb_importance_turing_model(
        problem.model, problem.polarization_power, problem.samples,
        toy_prior_model(problem.prior),
        observed, problem.frequencies, eff_psd, problem.observation_time,
        AnalyticInclination())
    marginalized = _marginalized_model(
        problem; effective_psd = eff_psd, amplitude_prior, grid, observed)

    θ = (; weight_shift = problem.theta.weight_shift)
    log_integrand = [Turing.logjoint(general, merge((; rate_scale = φ), θ)) for φ in grid]
    reference = _log_trapezoid(log_integrand, grid)

    @test Turing.logjoint(marginalized, θ) ≈ reference rtol = 1.0e-10
    # Anti-vacuity: the likelihood is doing real work at this σ, so the marginalized log
    # density is nowhere near the bare prior.
    @test !isapprox(Turing.logjoint(marginalized, θ),
        logpdf(problem.prior.weight_shift,
            θ.weight_shift); rtol = 1.0e-3)

    # And it tracks θ: a second shape point integrates to its own value.
    θ2 = (; weight_shift = 0.15)
    reference2 = _log_trapezoid(
        [Turing.logjoint(general, merge((; rate_scale = φ), θ2)) for φ in grid], grid)
    @test Turing.logjoint(marginalized, θ2) ≈ reference2 rtol = 1.0e-10
    @test !isapprox(reference, reference2)
end

@testset "the marginalized model's latents, := sites, and guards" begin
    problem = local_problem_context()
    model = _marginalized_model(problem)

    # The amplitude parameter is pinned *inside* the model, so it is not a latent -- and
    # it is not conditioned at the call site either, unlike every other fixed name.
    @test _varinfo_symbols(VarInfo(model)) == Set((:weight_shift,))

    θ = (; weight_shift = problem.theta.weight_shift)
    recorded = Turing.DynamicPPL.ParamsWithStats(
        Turing.DynamicPPL.InitFromParams(θ), model).params
    for name in AMPLITUDE_DERIVED
        @test haskey(recorded, @varname($name))
        @test isfinite(recorded[@varname($name)])
    end

    # `template_merger_rate` is the rate at the *pinned fiducial amplitude*, not at any
    # sampled φ: the model never publishes a number that reads as the physical rate.
    template_forward = forward_model(
        problem.model, problem.polarization_power, problem.samples,
        merge(problem.theta, (; rate_scale = problem.fiducials.rate_scale)))
    @test recorded[@varname(template_merger_rate)] == template_forward.rate

    # ρ = √(m|m) is the template's optimal SNR, by the repo's own SNR convention.
    obs_sec = year_to_second(problem.observation_time)
    df = frequency_bin_width(problem.frequencies)
    eff_psd = _signal_scale_psd(problem)
    @test recorded[@varname(template_optimal_snr)] ≈
          spectral_snr(template_forward.spectral_density, eff_psd, obs_sec, df)

    # `observed` was synthesized at the fiducials, and θ differs only in `weight_shift`,
    # so Â is close to but not exactly 1 -- close enough to prove it is a ratio, not a
    # rescaled inner product.
    @test 0.5 < recorded[@varname(amplitude_mle)] < 2.0

    # Sampling and marginalizing the same parameter is silent double-counting.
    both = _marginalized_model(problem; prior_model = toy_prior_model(problem.prior))
    @test_throws ArgumentError VarInfo(both)
end

@testset "the marginalized model samples" begin
    problem = local_problem_context()
    model = _marginalized_model(problem)
    chain = sample(
        model,
        Turing.NUTS(3, 0.8),
        3;
        progress = false,
        chain_type = FlexiChains.VNChain,
        initial_params = InitFromPrior()
    )
    @test sort(collect(Symbol.(FlexiChains.parameters(chain)))) ==
          sort(collect((:weight_shift, AMPLITUDE_DERIVED...)))
    @test all(isfinite, vec(Array(chain[:logjoint])))
end
