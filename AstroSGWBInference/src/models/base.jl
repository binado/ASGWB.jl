# The model contract is a documented callable, not a type or a generic function:
#
#     weights_fn(Λ, samples) -> (rate, log_weights)
#
# `rate` is the detector-frame merger rate in events per second; `log_weights` is one log
# importance weight per catalog sample. Anything callable with that signature qualifies --
# a functor carrying prepared caches, or a closure. See the `AstroSGWBInference` module
# docstring.

function _forward_model(
        weights_fn, fluxes, samples, Λ;
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    rate, log_weights = weights_fn(Λ, samples)
    weights = exp.(log_weights)
    Sh = AstroSGWB.spectral_density(fluxes, rate; weights = weights, average_mode)
    return (; rate, weights, spectral_density = Sh)
end

function _forward_spectral_density(
        weights_fn, fluxes, samples, Λ;
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    return _forward_model(weights_fn, fluxes, samples, Λ; average_mode).spectral_density
end

"""
    fiducial_spectral_density(weights_fn, fluxes, samples, fiducial_hyperparameters;
                              average_mode=AnalyticInclination()) -> Vector

Synthesize the observed strain spectral density at the fiducial hyperparameters using the
caller's `weights_fn(Λ, samples) -> (rate, log_weights)` callable.

`average_mode` is the inclination-averaging convention of the catalog that produced
`fluxes` (see [`AstroSGWB.spectral_density`](@ref)). It must match the mode used when
scoring the likelihood, or the synthesized data and the model disagree by a constant
factor; [`build_turing_model`](@ref) enforces that by forwarding a single value to both.
"""
function fiducial_spectral_density(
        weights_fn, fluxes, samples, fiducial_hyperparameters;
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    return _forward_spectral_density(
        weights_fn, fluxes, samples, fiducial_hyperparameters; average_mode)
end
