"""
    AstroSGWB

Astrophysical stochastic gravitational-wave background modeling: importance
sampling, redshift grids, and spectral-density forward models. Turing model construction lives in the
`AstroSGWBInference` package (see the `AstroSGWBInference/` directory in the repository).

The primary inference artifact is **`catalog.h5`** ([`WaveformCatalogFile`](@ref)):
per-sample intrinsic parameters with precomputed luminosity distances, and a
`(nfreq, nsamples)` per-sample flux matrix `|h_+|² + |h_×|²` (before the
fiducial `(D_L/D_gw)²` factor).

Callers define an importance adapter (or use one from `AstroSGWBImportanceModels`),
fiducial hyperparameters, and a catalog sample adapter in Julia, then pass raw catalog
`fluxes`, restructured `samples`, and fiducials explicitly. Prepared importance models
cache proposal log-probabilities, redshift interpolants, and rate metadata; detector
state lives separately in an [`ObservationContext`](@ref).

Inference state is a flat hyperparameter `NamedTuple`. The caller-owned model contract and
Turing integration live in `AstroSGWBInference`; this package provides the reusable physics
and array kernels used to implement that contract.
"""
module AstroSGWB

using CBCDistributions
using Cosmology
import CBCDistributions: single_event_prior
import Cosmology: cosmology, cosmology_type, gravitational_wave_distance,
                  gw_em_distance_ratio, hyperparameters,
                  propagation, propagation_type

include("types.jl")
include("catalog/grid.jl")
include("catalog/catalog.jl")
include("catalog/io.jl")
include("samples.jl")
include("detector/psd.jl")
include("detector/detector.jl")
include("detector/overlap.jl")
include("detector/effective_psd.jl")
include("detector/observation.jl")
include("spectral_density.jl")
include("snr.jl")
include("diagnostics.jl")

# Types
export ObservationContext,
       canonical_hyperparameters,
       validate_hyperparameters,
       CATALOG_SOURCE_TYPE_ATTR,
       CATALOG_SOURCE_TYPE_BNS,
       CumulativeIntegral1D,
       GridQuery,
       interpolate,
       cdf,
       normalizer,
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

# Detector network (ORF / PSD effective strain PSD; used by `build_observation_context`)
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
       LambdaCDM,
       W0CDM,
       W0WaCDM,
       GR,
       ModifiedPropagation,
       dark_energy_eos,
       de_density_ratio,
       cosmology,
       cosmology_type,
       SUPPORTED_COSMOLOGIES,
       propagation,
       propagation_hyperparameters,
       propagation_type,
       propagation_config_name,
       SUPPORTED_PROPAGATIONS,
       CosmologyCache,
       comoving_distance,
       luminosity_distance,
       luminosity_distance_at_sample,
       differential_comoving_volume,
       gravitational_wave_distance,
       gw_em_distance_ratio,
       hubble_constant_si,
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
       RedshiftInterpolatedDistribution

# Spectral density
export spectral_density,
       inner_product,
       spectral_snr_squared,
       spectral_snr,
       Ωgw

# Diagnostics
export normalized_ess

# Time conversions
export JULIAN_YEAR_SEC,
       year_to_second,
       second_to_year

end
