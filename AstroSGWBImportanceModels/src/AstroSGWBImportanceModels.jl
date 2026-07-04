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
                        GridQuery,
                        MadauDickinsonSourceFrame,
                        _normalized_log_density,
                        build_redshift_prior,
                        interpolate,
                        merger_rate_per_sec,
                        redshift_integral,
                        redshift_logpdf_eltype,
                        source_frame_distribution
import Cosmology
using Cosmology:
                 AbstractCosmology,
                 AbstractPropagation,
                 CosmologyCache,
                 cosmology,
                 gw_em_distance_ratio,
                 luminosity_distance,
                 luminosity_distance_at_sample,
                 propagation

export BNSMadauDickinsonImportanceModel,
       bns_madau_dickinson_hyperparameters,
       bns_samples_from_catalog,
       prepare_bns_madau_dickinson_model

include("models/bns_madau_dickinson_modified_propagation/bns_madau_dickinson.jl")

end
