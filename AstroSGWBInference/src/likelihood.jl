"""
    loglikelihood(Λ, weights_fn, fluxes, samples, observation::ObservationContext, observed;
                  average_mode=AnalyticInclination())

Gaussian in-band log-likelihood of the SGWB spectral density at `Λ`. Delegates the
cosmology-specific `rate`/`log_weights` to the caller's `weights_fn(Λ, samples)` callable,
exponentiates the weights, contracts the raw fluxes into `Sₕ`, and scores the in-band
residual against the `observation` masks/scales.

`observed` is the full-length strain spectral density vector (one entry per frequency bin
in `observation.frequencies`). `average_mode` is the inclination-averaging convention of
the catalog that produced `fluxes`; it must match the convention `observed` was built
under (see [`fiducial_spectral_density`](@ref)).
"""
function loglikelihood(
        Λ::NamedTuple,
        weights_fn,
        fluxes::AbstractMatrix{<:Real},
        samples::NamedTuple,
        observation::ObservationContext,
        observed::AbstractVector{<:Real};
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    Sh = _forward_spectral_density(weights_fn, fluxes, samples, Λ; average_mode)

    mask = observation.in_band_mask
    σ = observation.sgwb_scale_in_band
    residual = observed[mask] .- Sh[mask]
    return -0.5 * sum((residual ./ σ) .^ 2 .+ log.(2π .* (σ .^ 2)))
end

function logposterior(
        Λ::NamedTuple,
        weights_fn,
        fluxes::AbstractMatrix{<:Real},
        samples::NamedTuple,
        observation::ObservationContext,
        prior::ProductNamedTupleDistribution,
        observed::AbstractVector{<:Real};
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    return logpdf(prior, Λ) +
           loglikelihood(
        Λ, weights_fn, fluxes, samples, observation, observed; average_mode)
end
