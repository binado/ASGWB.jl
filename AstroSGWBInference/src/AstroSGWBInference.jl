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

Hyperparameter *names* are declared by the prior, not by the model: `keys(prior.dists)`
alone determines what is sampled. A name the callable needs but the prior omits surfaces
as a `KeyError` on `Λ.name` at the first evaluation, before the sampler burns wall clock.
"""
module AstroSGWBInference

include("InferenceImpl.jl")
using .InferenceImpl:
                      fiducial_spectral_density,
                      build_turing_model,
                      condition_turing_model,
                      loglikelihood,
                      logposterior,
                      AbstractAverageMode,
                      AnalyticInclination,
                      CatalogInclination

export fiducial_spectral_density,
       build_turing_model,
       condition_turing_model,
       loglikelihood,
       logposterior,
       AbstractAverageMode,
       AnalyticInclination,
       CatalogInclination,
       atomic_save_chain,
       MCMCConfig,
       SamplerConfig,
       load_config,
       save_config,
       validate_fiducials

include("chain_io.jl")
include("config.jl")
include("cli/stack_partial_chains.jl")
using .ChainIO: atomic_save_chain
using .Config: MCMCConfig, SamplerConfig, load_config, save_config, validate_fiducials

end
