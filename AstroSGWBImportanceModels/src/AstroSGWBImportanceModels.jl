"""
    AstroSGWBImportanceModels

Concrete, reusable importance-model adapters for `AstroSGWBInference`. The package owns
astrophysical model choices while `AstroSGWBInference` retains the generic two-method
model contract.
"""
module AstroSGWBImportanceModels

import AstroSGWBInference: hyperparameters, merger_rate_and_log_weights
using CBCDistributions:
                        DEFAULT_Z_GRID,
                        MadauDickinsonSourceFrame,
                        detector_frame_merger_rate_density,
                        merger_rate_per_sec,
                        source_frame_distribution
import Cosmology
using Cosmology:
                 AbstractCosmology,
                 AbstractPropagation,
                 GridInterpolator,
                 cosmology,
                 distance_and_volume_grid,
                 gw_em_distance_ratio,
                 luminosity_distance,
                 propagation,
                 trapz

export BNSMadauDickinsonImportanceModel,
       bns_madau_dickinson_hyperparameters,
       bns_samples_from_catalog,
       prepare_bns_madau_dickinson_model

include("models/bns_madau_dickinson.jl")

end
