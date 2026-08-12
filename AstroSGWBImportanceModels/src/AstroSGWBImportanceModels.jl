"""
    AstroSGWBImportanceModels

Concrete, reusable importance-model adapters for `AstroSGWBInference`. The package owns
astrophysical model choices; the inference package owns the sampler.

The seam between them is a **callable**, `merger_rate_and_log_weights_fn(Λ, samples) -> (rate, log_weights)`,
so prepared models here are functors and this package deliberately does **not** depend on
`AstroSGWBInference` -- nothing is imported from it and no methods are added to its
generics. The two-package split is a convenience, not a coupling.
"""
module AstroSGWBImportanceModels

using AstroSGWBDistributions:
                              DEFAULT_Z_GRID,
                              Interpolated1DDistribution,
                              MadauDickinsonSourceFrame,
                              RedshiftInterpolatedDistribution,
                              detector_frame_merger_rate_density,
                              normalizer,
                              source_frame_distribution
using AstroSGWBCosmology:
                          AbstractCosmology,
                          AbstractPropagation,
                          cosmology,
                          distance_and_volume_grid,
                          gw_em_distance_ratio,
                          propagation
using DataInterpolations: LinearInterpolation

export BNSMadauDickinsonImportanceModel,
       prepare_bns_madau_dickinson_model,
       AMPLITUDE_PARAMETERS,
       amplitude_H0,
       amplitude_R₀,
       merger_rate_amplitude_H0,
       merger_rate_amplitude_R₀,
       bns_amplitude_scalings

include("models/bns_madau_dickinson.jl")

end
