"""
    AstroSGWBInference

Turing/AdvancedHMC wrappers, the spectral-density forward model, and sampling helpers for
astrophysical stochastic gravitational-wave background inference.

# The model contract

Everything model-specific reaches this package through **one documented callable**:

    weights_fn(Λ, samples) -> (rate, log_weights)

- `Λ` is a flat `NamedTuple` of live hyperparameters.
- `samples` is the caller's per-event proposal sample collection.
- `rate` is the detector-frame merger rate in events per second.
- `log_weights` is one log importance weight per catalog sample.

There is no abstract type to subtype and no generic function to add methods to. A prepared
model is a **functor** -- a struct carrying its caches with a `(m::M)(Λ, samples)` method,
which keeps full dispatch and type parameters -- and an ad-hoc model is a plain closure:

    weights_fn = (Λ, samples) -> (1e-7 * Λ.rate_scale,
                                  fill(Λ.weight_shift, length(samples.redshift)))

Hyperparameter *names* are declared by the prior, not by the model: `keys(prior)`
determines the full hyperparameter set and the Turing variable creation order. Fixing a
hyperparameter is Turing conditioning — `model | (; R₀ = fiducials.R₀)` — so the chain
carries exactly the sampled variables by construction. A name the callable needs but the
prior omits surfaces as a `KeyError` on `Λ.name` at the first evaluation, before the
sampler burns wall clock.

[`forward_model`](@ref) is the single implementation of the forward pass, shared by the
`@model` body ([`astrosgwb_importance_turing_model`](@ref)) and the caller-side synthesis
of `observed` at the fiducial point. Scoring a point is `Turing.logjoint(model, θ)`;
there is deliberately no second likelihood implementation to drift from the first.
"""
module AstroSGWBInference

include("InferenceImpl.jl")
using .InferenceImpl:
                      forward_model,
                      astrosgwb_importance_turing_model,
                      AbstractAverageMode,
                      AnalyticInclination,
                      CatalogInclination

export forward_model,
       astrosgwb_importance_turing_model,
       AbstractAverageMode,
       AnalyticInclination,
       CatalogInclination,
       MCMCConfig,
       SamplerConfig,
       load_config,
       save_config

include("config.jl")
using .Config: MCMCConfig, SamplerConfig, load_config, save_config

end
