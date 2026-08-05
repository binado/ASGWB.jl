"""
    hyperparameters(model)

Return the complete collection of hyperparameter names used by `model`. Model authors
implement this method for their prepared model type. Must return a `Tuple{Vararg{Symbol}}`
of unique names; the order has no semantic meaning.
"""
function hyperparameters end

"""
    merger_rate_and_log_weights(model, Λ, samples) -> (rate, log_weights)

Evaluate the caller-owned, model-specific portion of the forward model. Implementations
return the detector-frame merger rate in events per second and one log importance weight
per catalog sample.
"""
function merger_rate_and_log_weights end

function _hyperparameter_names(model)
    names = hyperparameters(model)
    names isa Tuple{Vararg{Symbol}} || throw(
        ArgumentError(
        "hyperparameters(model) must return a Tuple of Symbols; got $(repr(names))",
    ),
    )
    length(unique(names)) == length(names) || throw(
        ArgumentError("hyperparameters(model) must contain unique names; got $(repr(names))"),
    )
    return names
end

function _validate_parameter_names(expected, actual; context::AbstractString)
    expected_set = Set(expected)
    actual_set = Set(actual)
    missing = sort!(collect(setdiff(expected_set, actual_set)); by = string)
    extra = sort!(collect(setdiff(actual_set, expected_set)); by = string)
    isempty(missing) && isempty(extra) && return nothing
    throw(ArgumentError(
        "$(context) parameter names do not match model hyperparameters; " *
        "missing=$(repr(Tuple(missing))), extra=$(repr(Tuple(extra)))",
    ))
end

function _forward_model(
        model, fluxes, samples, Λ;
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    rate, log_weights = merger_rate_and_log_weights(model, Λ, samples)
    weights = exp.(log_weights)
    Sh = AstroSGWB.spectral_density(fluxes, rate; weights = weights, average_mode)
    return (; rate, weights, spectral_density = Sh)
end

function _forward_spectral_density(
        model, fluxes, samples, Λ;
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    return _forward_model(model, fluxes, samples, Λ; average_mode).spectral_density
end

"""
    fiducial_spectral_density(model, fluxes, samples, fiducial_hyperparameters;
                              average_mode=AnalyticInclination()) -> Vector

Synthesize the observed strain spectral density at the fiducial hyperparameters using the
caller's [`merger_rate_and_log_weights`](@ref) implementation.

`average_mode` is the inclination-averaging convention of the catalog that produced
`fluxes` (see [`AstroSGWB.spectral_density`](@ref)). It must match the mode used when
scoring the likelihood, or the synthesized data and the model disagree by a constant
factor; [`build_turing_model`](@ref) enforces that by forwarding a single value to both.
"""
function fiducial_spectral_density(
        model, fluxes, samples, fiducial_hyperparameters;
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    return _forward_spectral_density(
        model, fluxes, samples, fiducial_hyperparameters; average_mode)
end
