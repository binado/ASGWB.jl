using Distributions: MvNormal
using LinearAlgebra: Diagonal
using Turing
using Turing: DynamicPPL

"""
    _prior_model_symbols(prior_model) -> Set{Symbol}

Top-level VarInfo symbols declared by a caller-owned prior `@model`. Used to reject
double-counting an amplitude-marginalized parameter.
"""
function _prior_model_symbols(prior_model)
    return Set(DynamicPPL.getsym(vn) for vn in keys(DynamicPPL.VarInfo(prior_model)))
end

"""
    gwbackground_importance_turing_model(merger_rate_and_log_weights_fn, polarization_power,
                                      samples, prior_model, observed, frequencies, effective_psd,
                                      observation_time, average_mode) -> DynamicPPL.Model

The Turing model scoring `merger_rate_and_log_weights_fn(Λ, samples) -> (rate, log_weights)`
against `observed` (see the `GWBackgroundInference` module docstring for the model contract).
There is no convenience constructor: callers with no external spectrum to fit synthesize
`observed` at the fiducial point themselves,
`forward_model(merger_rate_and_log_weights_fn, polarization_power, samples, fiducials; average_mode).spectral_density`.

**One `average_mode` must reach both sides.** The same value builds the data and scores
it; splitting them makes the synthesized `observed` and the model disagree by a constant
factor with no other symptom.

Every frequency bin is scored: `polarization_power`, `observed`, `frequencies`, and `effective_psd`
must already be restricted to the analysis band (slice them with one mask beforehand).
`effective_psd` is the network effective strain PSD from [`GWBackground.effective_psd`](@ref)
and `observation_time` the duration in years (Julian year); the per-bin Gaussian scale is
derived in the model body via [`GWBackground.gaussian_bin_scale`](@ref) from `effective_psd`,
`frequencies`, and `observation_time`.

`prior_model` is a caller-owned Turing `@model` whose `~` sites declare every hyperparameter
`merger_rate_and_log_weights_fn` reads (sampled or not). It is embedded with
`to_submodel(prior_model, false)` so conditioning and scoring use bare symbols. Fixing a
hyperparameter is Turing conditioning at the call site, `model | (; R₀ = fiducials.R₀)`:
the pinned value enters as an observation, its prior density folds into the joint as a
sampling-irrelevant constant, and the chain contains exactly the unconditioned variables
**by construction** -- no helpers, no subset validation. A pinned value outside its prior
support scores `-Inf` at the first evaluation, so a misconfigured pin fails loudly before
the sampler burns wall clock. A name the callable needs but `prior_model` omits surfaces as a
`KeyError` on `Λ.name` at the same point.

Two derived quantities are recorded on every saved draw as `:=` sites, so they reach the
chain (and the netCDF `posterior` group) rather than being computed and thrown away:
`total_merger_rate` and `importance_relative_ess`. The names and definitions match the
Python `astrogwb` stack's `spectral_density_model`, so the two projects' netCDFs are
directly comparable. `:=` is free here: its right-hand side is evaluated unconditionally
either way, and Turing already re-evaluates the model once per saved draw to collect the
log probabilities in `sample_stats`.

`average_mode` is positional rather than a keyword: the positional path through DynamicPPL
is what the tests exercise, and positional arguments stay visible in `model.args` when
introspecting a built model. Singleton instances (not `Type`s) pass through
`transform_args` untouched.
"""
@model function gwbackground_importance_turing_model(
        merger_rate_and_log_weights_fn,
        polarization_power::AbstractMatrix{<:Real},
        samples::NamedTuple,
        prior_model,
        observed::AbstractVector{<:Real},
        frequencies::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        observation_time::Real,
        average_mode::AbstractAverageMode
)
    # `false`: no varname prefixing, so caller-side conditioning (`model | (; R₀ = …)`)
    # and scoring (`Turing.logjoint(model, θ)`) address the prior model's variables by the
    # same bare symbols the caller already uses.
    Λ ~ to_submodel(prior_model, false)
    forward = forward_model(merger_rate_and_log_weights_fn, polarization_power, samples,
        Λ; average_mode)
    Sh = forward.spectral_density

    # O(nfreq) elementwise work -- noise next to the weight contraction.
    obs_sec = year_to_second(observation_time)
    scale = gaussian_bin_scale(;
        effective_psd = effective_psd,
        frequencies = frequencies,
        observation_time_sec = obs_sec)

    total_merger_rate := forward.rate
    importance_relative_ess := normalized_ess(forward.weights)

    observed ~ MvNormal(
        Sh,
        Diagonal(scale .^ 2)
    )
    return nothing
end

"""
    gwbackground_amplitude_marginalized_turing_model(merger_rate_and_log_weights_fn,
                                                  polarization_power, samples, prior_model,
                                                  observed, frequencies, effective_psd,
                                                  observation_time, average_mode,
                                                  amplitude_fiducial, amplitude_fn,
                                                  amplitude_prior, amplitude_grid)
        -> DynamicPPL.Model

Identical to [`gwbackground_importance_turing_model`](@ref) except that one strictly
multiplicative hyperparameter is **integrated out** of the Gaussian likelihood instead of
being sampled. That removes the long curved amplitude--shape degeneracy NUTS handles
worst, and it is nearly free here: the amplitude never touches the importance weights and
`polarization_power` is a fixed catalog.

The marginalized parameter is named by `amplitude_fiducial`, a **one-entry** `NamedTuple`
such as `(; H0 = 67.66)` -- a NamedTuple rather than a `Symbol => value` pair so the name
is a compile-time constant and the `merge` below stays type-stable. Its value is the
reference ``\\varphi_\\mathrm{fid}`` that defines the template. `amplitude_fn` is the
absolute scaling ``f = g_R g_F``, `amplitude_prior` the prior ``\\pi(\\varphi)``, and
`amplitude_grid` the quadrature nodes (see [`quadrature_grid`](@ref)). For the canonical
BNS adapter these come from `GWBackgroundImportanceModels.bns_amplitude_scalings`.

Unlike every other fixed hyperparameter, the amplitude parameter is pinned **inside the
model** (`merge(Λ, amplitude_fiducial)`), not by conditioning at the call site — the model
needs the template spectrum at the fiducial to define the amplitude ratio at all. It must
therefore *not* also appear in `prior_model`, which would be silent double-counting; that is an
`ArgumentError` at the first evaluation.

Recorded as `:=` sites: `template_merger_rate`, `amplitude_mle`, `template_optimal_snr`,
`importance_relative_ess`. `template_merger_rate` is the rate at the pinned fiducial
amplitude -- the model never publishes a number that would be mistaken for the physical
merger rate at an unmarginalized ``\\varphi``; [`reconstruct_amplitude`](@ref) recovers
that in post-processing, along with ``\\varphi`` itself, from `amplitude_mle` and
`template_optimal_snr`. Those two are carried rather than the raw inner products because
they are better conditioned, directly interpretable (``\\sigma_A = 1/\\rho``), and
invertible by multiplication alone: ``(m|m) = \\rho^2`` and ``(d|m) = \\hat A \\rho^2``.

There is no `observed ~` site: the likelihood enters through `@addlogprob!` as

    log p(d | Â, θ) + log Z,

the Gaussian log-likelihood at the MLE amplitude plus the conditional's log normalizer.
`data_norm` is kept in that expression even though it is θ-independent, so the absolute
log density matches [`gwbackground_importance_turing_model`](@ref) exactly rather than up to
a constant.

The three σ-space contractions go through [`GWBackground.inner_product`](@ref), which already
carries the `2 T Δf` prefactor that makes it consistent with
[`GWBackground.gaussian_bin_scale`](@ref); a PSD-space contraction would make ``\\rho^2`` too
small by `2T`.
"""
@model function gwbackground_amplitude_marginalized_turing_model(
        merger_rate_and_log_weights_fn,
        polarization_power::AbstractMatrix{<:Real},
        samples::NamedTuple,
        prior_model,
        observed::AbstractVector{<:Real},
        frequencies::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        observation_time::Real,
        average_mode::AbstractAverageMode,
        amplitude_fiducial::NamedTuple,
        amplitude_fn,
        amplitude_prior::Distributions.ContinuousUnivariateDistribution,
        amplitude_grid::AbstractVector{<:Real}
)
    # Sampling and marginalizing the same parameter double-counts it with no visible
    # symptom, so it is rejected on the first evaluation rather than silently tolerated.
    amp_name = first(keys(amplitude_fiducial))
    amp_name in _prior_model_symbols(prior_model) && throw(ArgumentError(
        "$(repr(amp_name)) is marginalized analytically and " *
        "cannot also be sampled; remove it from `prior_model`",
    ))

    Λ ~ to_submodel(prior_model, false)
    # The amplitude parameter is pinned here, inside the model: what the callable returns
    # is the *template* m(θ), and the marginalized amplitude is the ratio to it.
    Λ_template = merge(Λ, amplitude_fiducial)
    forward = forward_model(merger_rate_and_log_weights_fn, polarization_power, samples,
        Λ_template; average_mode)
    m = forward.spectral_density

    df = frequency_bin_width(frequencies)
    obs_sec = year_to_second(observation_time)
    scale = gaussian_bin_scale(;
        effective_psd = effective_psd,
        frequencies = frequencies,
        observation_time_sec = obs_sec)

    template_norm = inner_product(m, m, effective_psd, obs_sec, df)
    data_template = inner_product(observed, m, effective_psd, obs_sec, df)
    data_norm = inner_product(observed, observed, effective_psd, obs_sec, df)

    template_merger_rate := forward.rate
    amplitude_mle := data_template / template_norm
    template_optimal_snr := sqrt(template_norm)
    importance_relative_ess := normalized_ess(forward.weights)

    conditional = AmplitudeConditional(
        amplitude_mle,
        template_optimal_snr;
        amplitude_fn = amplitude_fn,
        prior = amplitude_prior,
        fiducial = only(amplitude_fiducial),
        grid = amplitude_grid
    )

    # log p(d | Â, θ) = normalization - ½ χ²(Â, θ) with χ²(Â, θ) = (d|d) - Â (d|m), the
    # completed square. Reuses `amplitude_mle` and `data_template` instead of re-forming
    # the (nfreq,) residual d - Â m through a second Gaussian evaluation.
    normalization = -sum(log.(scale) .+ 0.5 * log(2π))
    log_likelihood_at_mle = normalization -
                            0.5 * (data_norm - amplitude_mle * data_template)
    # The marginalization factor *is* the normalizing constant of the conditional that
    # `reconstruct_amplitude` later samples.
    Turing.@addlogprob! (log_likelihood_at_mle + log_normalizer(conditional))
    return nothing
end
