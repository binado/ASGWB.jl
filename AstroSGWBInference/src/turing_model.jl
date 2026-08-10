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

# `average_mode` is positional rather than a keyword: `track::Bool` already
# proves the positional path through DynamicPPL here, the model is unexported
# with a single caller, and a positional argument stays visible in `model.args`
# when introspecting a built model. Singleton instances (not `Type`s) pass
# through `transform_args` untouched.
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
    Λ_sampled ~ to_submodel(sample_hyperparameters(keys(prior), prior), false)
    # `merge(constants, Λ_sampled)` is the idiomatic Julia `{**constants, **sampled}`:
    # resolved at compile time on `NamedTuple`s, so it costs nothing per evaluation and
    # keeps `Λ.γ` type-stable. Sampled values win on collision, which `build_turing_model`
    # rejects up front rather than allowing a silently shadowed constant.
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
        snr,
    )
end

"""
    build_turing_model(weights_fn, polarization_power, samples, fiducial_hyperparameters,
                       frequencies, effective_psd, observation_time, prior;
                       constants=NamedTuple(), track=false, observed=nothing,
                       average_mode=AnalyticInclination())

Build the Turing model scoring `weights_fn` against `observed` (synthesized at
`fiducial_hyperparameters` when omitted). `weights_fn(Λ, samples) -> (rate, log_weights)`
is the whole model contract; see the `AstroSGWBInference` module docstring.

Every frequency bin is scored: `polarization_power`, `observed`, `frequencies`, and `effective_psd`
must already be restricted to the analysis band (slice them with one mask beforehand).
`effective_psd` is the network effective strain PSD from [`AstroSGWB.effective_psd`](@ref)
and `observation_time` the duration in years (Julian year); the per-bin Gaussian scale is
derived in the model body via [`AstroSGWB.gaussian_bin_scale`](@ref) from `effective_psd`,
`frequencies`, and `observation_time`, so the likelihood σ and the SNR tracking branch
always share one noise convention.

`prior` declares what is **sampled** and `constants` declares what is **fixed**; the model
body evaluates at `merge(constants, Λ_sampled)`. To sample a subset, build the prior with
only that subset and pass the rest as `constants` -- the chain then contains exactly the
sampled variables **by construction**, with no DynamicPPL conditioning and no subset
validation. A key present in both is rejected here rather than silently shadowed.

`fiducial_hyperparameters` is the full point (`prior ∪ constants`) at which `observed` is
synthesized, and stays a separate argument for exactly that reason.

`keys(prior)` alone declares which hyperparameters are sampled and in what order. A key
the model needs but neither `prior` nor `constants` supplies surfaces as a `KeyError` on
`Λ.name` at the first evaluation, before the sampler burns wall clock.
"""
function build_turing_model(
        weights_fn,
        polarization_power::AbstractMatrix{<:Real},
        samples::NamedTuple,
        fiducial_hyperparameters::NamedTuple,
        frequencies::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        observation_time::Real,
        prior::NamedTuple;
        constants::NamedTuple = NamedTuple(),
        track::Bool = false,
        observed::Union{Nothing, AbstractVector{<:Real}} = nothing,
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    # A check on *this call's two arguments*, not model-declared name bookkeeping:
    # `merge` lets the sampled value win, so an overlapping key would silently ignore
    # the constant the caller asked for.
    overlap = intersect(keys(prior), keys(constants))
    isempty(overlap) ||
        throw(ArgumentError("constants and prior both declare $(Tuple(overlap))"))

    # One `average_mode` reaches both the synthesized `observed` and the model
    # that scores it. Splitting them would bias the fit by a constant factor
    # with no other symptom, so they are deliberately not separately settable.
    observed_data = if observed === nothing
        forward_model(
            weights_fn, polarization_power, samples, fiducial_hyperparameters;
            average_mode).spectral_density
    else
        observed
    end
    return astrosgwb_importance_turing_model(
        track,
        average_mode,
        weights_fn,
        polarization_power,
        samples,
        frequencies,
        effective_psd,
        observation_time,
        prior,
        constants,
        observed_data
    )
end
