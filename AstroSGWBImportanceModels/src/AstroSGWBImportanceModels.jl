"""
    AstroSGWBImportanceModels

Concrete, reusable importance-model adapters for `AstroSGWBInference`. The package owns
astrophysical model choices; the inference package owns the sampler.

The seam between them is a **callable**, `weights_fn(Λ, samples) -> (rate, log_weights)`,
so prepared models here are functors and this package deliberately does **not** depend on
`AstroSGWBInference` -- nothing is imported from it and no methods are added to its
generics. The two-package split is a convenience, not a coupling.
"""
module AstroSGWBImportanceModels

using CBCDistributions:
                        DEFAULT_Z_GRID,
                        MadauDickinsonSourceFrame,
                        detector_frame_merger_rate_density,
                        merger_rate_per_sec,
                        source_frame_distribution
using Cosmology:
                 AbstractCosmology,
                 AbstractPropagation,
                 cosmology,
                 distance_and_volume_grid,
                 gw_em_distance_ratio,
                 luminosity_distance,
                 propagation,
                 trapz,
                 validate_redshift_grid
using DataInterpolations: LinearInterpolation

export BNSMadauDickinsonImportanceModel,
       bns_samples_from_catalog,
       prepare_bns_madau_dickinson_model

include("models/bns_madau_dickinson.jl")

end
