"""
    ASGWB

Astrophysical stochastic gravitational-wave background modeling: importance
sampling, redshift grids, and likelihoods. Turing model construction lives in the
`ASGWBInference` package (see the `ASGWBInference/` directory in the repository).

The primary inference artifact is **`catalog.h5`** ([`WaveformCatalogFile`](@ref)):
per-sample intrinsic parameters with precomputed luminosity distances, and a
`(n_freq, n_samples)` per-sample flux matrix `|h_+|² + |h_×|²` (before the
fiducial `(D_L/D_gw)²` factor).

Callers define their population model, fiducial hyperparameters, and catalog sample
adapter in Julia, then construct a pure [`ImportanceSamplingProblem`](@ref). Derived
`Λ`-independent caches (rescaled fluxes, proposal log-prob, redshift interpolant,
detector PSDs, fiducial spectral density) are built into a [`ModelContext`](@ref) by
[`build_model_context`](@ref).

Inference state is a flat hyperparameter `NamedTuple` validated against the
[`PopulationModel`](@ref) contract; the cosmology family `C` is threaded through atomic
calls rather than stored on the problem.
"""
module ASGWB

using CBCDistributions
import CBCDistributions: cosmology, cosmology_type, gravitational_wave_distance,
                         hyperparameters, single_event_prior

include("types.jl")
include("models/base.jl")
include("catalog/grid.jl")
include("catalog/catalog.jl")
include("catalog/io.jl")
include("inference_types.jl")
include("detector/psd.jl")
include("detector/detector.jl")
include("detector/overlap.jl")
include("detector/effective_psd.jl")
include("detector/observation.jl")
include("importance.jl")
include("spectral_density.jl")
include("snr.jl")
include("diagnostics.jl")
include("context.jl")
include("posterior.jl")

# Types
export ImportanceSamplingProblem,
       ModelContext,
       build_model_context,
       ObservationContext,
       PopulationModel,
       hyperparameters,
       single_event_prior,
       full_hyperparameters,
       canonical_hyperparameters,
       validate_hyperparameters,
       validate_subset,
       stack_source_masses,
       CATALOG_SOURCE_TYPE_ATTR,
       CATALOG_SOURCE_TYPE_BNS,
       CumulativeIntegral1D,
       RedshiftPrior,
       redshift

# Catalog I/O
export FrequencyGrid,
       frequencies,
       in_band_mask,
       WaveformCatalog,
       WaveformCatalogMetadata,
       WaveformCatalogFile,
       load_catalog,
       save_catalog

# Detector network (ORF / PSD effective strain PSD; used by `build_model_context`)
export Detector,
       PowerSpectralDensity,
       default_detector_data_dir,
       overlap_reduction_function,
       pairwise_overlap_reduction_function,
       effective_psd,
       gaussian_bin_scale,
       gaussian_bin_variance,
       frequency_bin_width,
       build_observation_context

# Cosmology
export E,
       AbstractCosmology,
       LambdaCDM,
       W0CDM,
       W0WaCDM,
       ModifiedPropagation,
       dark_energy_eos,
       de_density_ratio,
       cosmology,
       cosmology_config_name,
       cosmology_type,
       SUPPORTED_COSMOLOGIES,
       CosmologyCache,
       comoving_distance,
       luminosity_distance,
       differential_comoving_volume,
       gravitational_wave_distance,
       H0,
       Ωm

# Redshift & population
export madau_dickinson_source_frame_distribution,
       detector_frame_merger_rate_density,
       build_redshift_prior,
       redshift_prior,
       MadauDickinsonSourceFrame,
       source_frame_distribution,
       DEFAULT_Z_GRID,
       redshift_log_prob,
       redshift_logpdf_eltype,
       redshift_integral,
       expected_number_of_events,
       merger_rate_per_sec

# Priors
export OrderedUniformSourceMassPair,
       AlignedSpinChiSimple,
       BNS_LAMBDA_HIGH,
       RedshiftInterpolatedDistribution,
       batched_logpdf

# Importance sampling
export importance_weights,
       compute_importance_weights,
       spectral_density,
       inner_product,
       spectral_snr_squared,
       spectral_snr,
       Ωgw

# Diagnostics
export normalized_ess, max_normalized_weight, log_ratio_variance

# Posterior
export loglikelihood,
       merger_rate,
       fiducial_hyperparameters,
       fiducial_redshift_integral

end
