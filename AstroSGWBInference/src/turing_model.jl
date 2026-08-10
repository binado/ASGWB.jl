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
                                      prior, constants, observed) -> DynamicPPL.Model

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

`prior` declares what is **sampled** and `constants` declares what is **fixed**; the model
body evaluates at `merge(constants, Λ_sampled)`. To sample a subset, build the prior with
only that subset and pass the rest as `constants` -- the chain then contains exactly the
sampled variables **by construction**, with no DynamicPPL conditioning and no subset
validation. A key present in both throws an `ArgumentError` at the first evaluation
rather than silently shadowing the constant.

When `track` is true, each evaluation also returns `(; number_of_sources,
effective_sample_size, snr)`.

`keys(prior)` alone declares which hyperparameters are sampled and in what order. A key
the model needs but neither `prior` nor `constants` supplies surfaces as a `KeyError` on
`Λ.name` at the first evaluation, before the sampler burns wall clock.

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
        constants::NamedTuple,
        observed::AbstractVector{<:Real}
)
    # `merge` lets the sampled value win on collision, so an overlapping key would
    # silently ignore the constant the caller asked for. Reject it at the first
    # evaluation -- the same point where a missing name surfaces as a `KeyError` on `Λ`.
    overlap = intersect(keys(prior), keys(constants))
    isempty(overlap) ||
        throw(ArgumentError("constants and prior both declare $(Tuple(overlap))"))

    Λ_sampled ~ to_submodel(sample_hyperparameters(keys(prior), prior), false)
    # `merge(constants, Λ_sampled)` is the idiomatic Julia `{**constants, **sampled}`:
    # resolved at compile time on `NamedTuple`s, so it costs nothing per evaluation and
    # keeps `Λ.γ` type-stable. Sampled values win on collision; the check above rejects
    # collisions rather than allowing a silently shadowed constant.
    Λ = merge(constants, Λ_sampled)
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
