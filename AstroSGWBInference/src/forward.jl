"""
    forward_model(weights_fn, polarization_power, samples, Λ; average_mode=AnalyticInclination())
        -> (; rate, weights, spectral_density)

Evaluate the forward model at hyperparameters `Λ`: call the model contract
`weights_fn(Λ, samples) -> (rate, log_weights)`, exponentiate the weights, and contract
the raw `(nfreq, nsamples)` polarization-power matrix into the strain spectral density `Sₕ`.

This is the **only** implementation of the forward pass. The `@model` body and the
synthesis of `observed` from the fiducial point both go through it, so the two cannot
drift apart -- a second implementation of one likelihood is a parity-bug generator, and
`Turing.logjoint(model, θ)` already covers everything a bespoke `logposterior` did.

`rate` and `weights` are returned alongside `spectral_density` because the tracked branch
of the Turing model reports `number_of_sources` and `effective_sample_size` from them;
callers wanting only the spectrum take `.spectral_density`.

`average_mode` is the inclination-averaging convention of the catalog that produced
`polarization_power` (see [`AstroSGWB.spectral_density`](@ref)). It must match the mode `observed` was
built under, or the synthesized data and the model that scores it disagree by a constant
factor; callers pass a single value to both this call and the Turing model.
"""
function forward_model(
        weights_fn, polarization_power, samples, Λ;
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    rate, log_weights = weights_fn(Λ, samples)
    weights = exp.(log_weights)
    return (; rate, weights,
        spectral_density = AstroSGWB.spectral_density(
            polarization_power, rate; weights, average_mode))
end
