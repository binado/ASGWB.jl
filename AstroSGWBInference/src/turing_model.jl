using Distributions: MvNormal
using LinearAlgebra: Diagonal
using Turing
using Turing: DynamicPPL

@model function sample_hyperparameters(order::Tuple{Vararg{Symbol}}, dists)
    values = map(order) do sym
        x ~ DynamicPPL.NamedDist(dists[sym], sym)
        x
    end
    return NamedTuple{order}(Tuple(values))
end

"""
    astrosgwb_importance_turing_model(track, average_mode, weights_fn, polarization_power,
                                      samples, frequencies, effective_psd, observation_time,
                                      prior, observed) -> DynamicPPL.Model

The Turing model scoring `weights_fn(Λ, samples) -> (rate, log_weights)` against
`observed` (see the `AstroSGWBInference` module docstring for the model contract). There
is no convenience constructor: callers with no external spectrum to fit synthesize
`observed` at the fiducial point themselves,
`forward_model(weights_fn, polarization_power, samples, fiducials; average_mode).spectral_density`.

**One `average_mode` must reach both sides.** The same value builds the data and scores
it; splitting them makes the synthesized `observed` and the model disagree by a constant
factor with no other symptom.

Every frequency bin is scored: `polarization_power`, `observed`, `frequencies`, and `effective_psd`
must already be restricted to the analysis band (slice them with one mask beforehand).
`effective_psd` is the network effective strain PSD from [`AstroSGWB.effective_psd`](@ref)
and `observation_time` the duration in years (Julian year); the per-bin Gaussian scale is
derived in the model body via [`AstroSGWB.gaussian_bin_scale`](@ref) from `effective_psd`,
`frequencies`, and `observation_time`, so the likelihood σ and the `track = true` SNR
always share one noise convention.

`prior` declares **every** hyperparameter `weights_fn` reads, sampled or not; `keys(prior)`
controls the Turing variable creation order. Fixing a hyperparameter is Turing
conditioning at the call site, `model | (; R₀ = fiducials.R₀)`: the pinned value enters
as an observation, its prior density folds into the joint as a sampling-irrelevant
constant, and the chain contains exactly the unconditioned variables **by construction** --
no helpers, no subset validation. A pinned value outside its prior support scores `-Inf`
at the first evaluation, so a misconfigured pin fails loudly before the sampler burns wall
clock. A name the callable needs but `prior` omits surfaces as a `KeyError` on `Λ.name` at
the same point.

When `track` is true, each evaluation also returns `(; number_of_sources,
effective_sample_size, snr)`.

`track` and `average_mode` are positional rather than keywords: the positional path
through DynamicPPL is what the tests exercise, and positional arguments stay visible in
`model.args` when introspecting a built model. Singleton instances (not `Type`s) pass
through `transform_args` untouched.
"""
@model function astrosgwb_importance_turing_model(
        track::Bool,
        average_mode::AbstractAverageMode,
        weights_fn,
        polarization_power::AbstractMatrix{<:Real},
        samples::NamedTuple,
        frequencies::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        observation_time::Real,
        prior::NamedTuple,
        observed::AbstractVector{<:Real}
)
    # `false`: no varname prefixing, so caller-side conditioning (`model | (; R₀ = …)`)
    # and scoring (`Turing.logjoint(model, θ)`) address the submodel's variables by the
    # same bare symbols the caller already uses.
    Λ ~ to_submodel(sample_hyperparameters(keys(prior), prior), false)
    forward = forward_model(weights_fn, polarization_power, samples, Λ; average_mode)
    Sh = forward.spectral_density

    # Derived from the same `effective_psd`, `frequencies`, and `observation_time` the
    # tracking branch reads, so the likelihood σ and the SNR convention are identical by
    # construction. O(nfreq) elementwise work -- noise next to the weight contraction.
    df = frequency_bin_width(frequencies)
    obs_sec = year_to_second(observation_time)
    scale = gaussian_bin_scale(;
        effective_psd = effective_psd,
        frequencies = frequencies,
        observation_time_sec = obs_sec)
    observed ~ MvNormal(
        Sh,
        Diagonal(scale .^ 2)
    )

    track || return nothing
    snr = spectral_snr(Sh, effective_psd, obs_sec, df)
    return (;
        number_of_sources = forward.rate * obs_sec,
        effective_sample_size = normalized_ess(forward.weights),
        snr
    )
end
