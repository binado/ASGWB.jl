"""
    AstroSGWBInference

Turing/AdvancedHMC wrappers, the spectral-density forward model, and sampling helpers for
astrophysical stochastic gravitational-wave background inference.

# The model contract

Everything model-specific reaches this package through **one documented callable**:

    merger_rate_and_log_weights_fn(Λ, samples) -> (rate, log_weights)

- `Λ` is a flat `NamedTuple` of live hyperparameters.
- `samples` is the caller's per-event proposal sample collection.
- `rate` is the detector-frame merger rate in events per second.
- `log_weights` is one log importance weight per catalog sample.

There is no abstract type to subtype and no generic function to add methods to. A prepared
model is a **functor** -- a struct carrying its caches with a `(m::M)(Λ, samples)` method,
which keeps full dispatch and type parameters -- and an ad-hoc model is a plain closure:

    merger_rate_and_log_weights_fn = (Λ, samples) -> (1e-7 * Λ.rate_scale,
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

# Two likelihoods

[`astrosgwb_amplitude_marginalized_turing_model`](@ref) integrates one strictly
multiplicative hyperparameter out of the Gaussian likelihood instead of sampling it,
removing the amplitude--shape degeneracy NUTS handles worst. It publishes the amplitude
sufficient statistics, and [`reconstruct_amplitude`](@ref) turns them back into draws of
the physical parameter in post-processing. See `amplitude.jl` for the math.

# Writing chains

[`NETCDF_PARAMETER_NAMES`](@ref) maps the Unicode hyperparameter names used everywhere in
Julia code, config, and tests onto ASCII names **on netCDF write only**, so the files this
package produces carry the same variable names as the Python `astrogwb` stack's. Applying
it is [`rename_posterior_for_netcdf`](@ref), which — together with
[`merge_into_posterior`](@ref) — lives in a package extension and requires
`InferenceObjects` to be loaded.
"""
module AstroSGWBInference

include("InferenceImpl.jl")
using .InferenceImpl:
                      forward_model,
                      astrosgwb_importance_turing_model,
                      astrosgwb_amplitude_marginalized_turing_model,
                      AmplitudeConditional,
                      quadrature_grid,
                      log_normalizer,
                      effective_nodes,
                      reconstruct_amplitude,
                      AbstractAverageMode,
                      AnalyticInclination,
                      CatalogInclination

export forward_model,
       astrosgwb_importance_turing_model,
       astrosgwb_amplitude_marginalized_turing_model,
       AmplitudeConditional,
       quadrature_grid,
       log_normalizer,
       effective_nodes,
       reconstruct_amplitude,
       AbstractAverageMode,
       AnalyticInclination,
       CatalogInclination,
       MCMCConfig,
       SamplerConfig,
       load_config,
       save_config,
       posterior_params,
       NETCDF_PARAMETER_NAMES,
       rename_posterior_for_netcdf,
       merge_into_posterior

include("config.jl")
using .Config: MCMCConfig, SamplerConfig, load_config, save_config, posterior_params,
               NETCDF_PARAMETER_NAMES

"""
    rename_posterior_for_netcdf(idata::InferenceData) -> InferenceData

Rename the `posterior` group's variables through [`NETCDF_PARAMETER_NAMES`](@ref),
immediately before `InferenceObjects.to_netcdf`.

Unicode hyperparameter names (`Ωm`, `Ξ₀`, `γ`, …) stay put in Julia code, config TOML, and
tests, where they are the physics notation the rest of the repo reads in; only the written
file gets ASCII names, which is what makes a Julia netCDF and a Python `astrogwb` netCDF
diffable variable-for-variable.

Requires `InferenceObjects` to be loaded (this method lives in a package extension).
"""
function rename_posterior_for_netcdf end

"""
    merge_into_posterior(idata::InferenceData, nt::NamedTuple) -> InferenceData

Add the variables in `nt` to `idata`'s `posterior` group, reusing the group's own `draw`
and `chain` dimensions so the new arrays align with the sampled ones by construction.
Used to fold [`reconstruct_amplitude`](@ref)'s outputs into the chain before writing.

Requires `InferenceObjects` to be loaded (this method lives in a package extension).
"""
function merge_into_posterior end

end
