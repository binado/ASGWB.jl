using Distributions: MvNormal
using LinearAlgebra: Diagonal
using Turing
using Turing: DynamicPPL

function _validate_subset(subset::Tuple{Vararg{Symbol}}, order)
    for symbol in subset
        symbol in order ||
            throw(ArgumentError("subset contains $(repr(symbol)); expected symbols from $(Tuple(order))"))
    end
    length(unique(subset)) == length(subset) ||
        throw(ArgumentError("subset must not repeat symbols"))
    return subset
end

function condition_turing_model(
        turing_model,
        theta0::NamedTuple,
        prior::NamedTuple,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}}
)
    order = keys(prior)
    sample_only === nothing && return turing_model
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the argument or pass `nothing` to sample every hyperparameter",
    ),
    )
    _validate_subset(sample_only, order)
    fixed = Tuple(s for s in order if s ∉ sample_only)
    isempty(fixed) && return turing_model
    return turing_model | (; (s => theta0[s] for s in fixed)...)
end

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
        fluxes::AbstractMatrix{<:Real},
        samples::NamedTuple,
        observation::ObservationContext,
        prior::NamedTuple,
        observed_in_band::AbstractVector{<:Real}
)
    Λ ~ to_submodel(sample_hyperparameters(keys(prior), prior), false)
    forward = _forward_model(weights_fn, fluxes, samples, Λ; average_mode)
    Sh = forward.spectral_density

    observed_in_band ~ MvNormal(
        Sh[observation.in_band_mask],
        Diagonal(observation.sgwb_scale_in_band .^ 2)
    )

    track || return nothing
    m = observation.in_band_mask
    df = frequency_bin_width(observation.frequencies)
    obs_sec = year_to_second(observation.observation_time)
    snr_sq = spectral_snr_squared(
        Sh[m], observation.effective_psd[m], obs_sec, df)
    return (;
        number_of_sources = forward.rate * obs_sec,
        effective_sample_size = normalized_ess(forward.weights),
        spectral_snr_squared = snr_sq,
        spectral_snr = sqrt(snr_sq)
    )
end

"""
    build_turing_model(weights_fn, fluxes, samples, fiducial_hyperparameters,
                       observation, prior; track=false, observed=nothing,
                       average_mode=AnalyticInclination())

Build the Turing model scoring `weights_fn` against `observed` (synthesized at
`fiducial_hyperparameters` when omitted). `weights_fn(Λ, samples) -> (rate, log_weights)`
is the whole model contract; see the `AstroSGWBInference` module docstring.

`keys(prior)` alone declares which hyperparameters are sampled and in what order.
A key the model needs but the prior omits surfaces as a `KeyError` on `Λ.name` at the
first evaluation, before the sampler burns wall clock.
"""
function build_turing_model(
        weights_fn,
        fluxes::AbstractMatrix{<:Real},
        samples::NamedTuple,
        fiducial_hyperparameters::NamedTuple,
        observation::ObservationContext,
        prior::NamedTuple;
        track::Bool = false,
        observed::Union{Nothing, AbstractVector{<:Real}} = nothing,
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    # One `average_mode` reaches both the synthesized `observed` and the model
    # that scores it. Splitting them would bias the fit by a constant factor
    # with no other symptom, so they are deliberately not separately settable.
    observed_data = if observed === nothing
        fiducial_spectral_density(
            weights_fn, fluxes, samples, fiducial_hyperparameters; average_mode)
    else
        observed
    end
    return astrosgwb_importance_turing_model(
        track,
        average_mode,
        weights_fn,
        fluxes,
        samples,
        observation,
        prior,
        observed_data[observation.in_band_mask]
    )
end
