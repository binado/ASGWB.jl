"""
    AstroSGWB

Astrophysical stochastic gravitational-wave background modeling: importance
sampling, redshift grids, and spectral-density forward models. Turing model construction lives in the
`AstroSGWBInference` package (see the `AstroSGWBInference/` directory in the repository).

The primary inference artifact is a **`waveform_catalog` v1 HDF5 file**, read by
[`load_catalog`](@ref) into an [`SGWBCatalog`](@ref): per-sample source parameters,
the shared frequency axis and in-band mask, and a `(nfreq, nsamples)` per-sample
polarization-power matrix `|h_+|² + |h_×|²` reduced from the stored complex polarizations.
Format IO lives in `PlusCross.jl`, so the same file is consumed unchanged by the
Python `astrogwb` package.

As loaded the polarization-power matrix is referenced to the **electromagnetic** luminosity distance
(before the fiducial `(D_L/D_gw)²` factor). Call
[`apply_gw_distance_correction!`](@ref) at the fiducial propagation before preparing an
importance model; the log-weights carry the compensating `+2 log Ξ_fid` term
unconditionally, so skipping the call under a non-GR fiducial biases the fit by `Ξ_fid²`.
The correction is **not idempotent**: an in-place call mutates the catalog it is given,
so reactive/Pluto call sites should use the out-of-place
[`apply_gw_distance_correction`](@ref).

The catalog also carries the inclination-averaging convention: [`average_mode`](@ref)
derives it from the `inclination` column, and it must be threaded to
[`spectral_density`](@ref) and to Turing model construction.

Callers define an importance adapter (or use one from `AstroSGWBImportanceModels`),
fiducial hyperparameters, and a catalog sample adapter in Julia, then pass raw catalog
`polarization_power`, restructured `samples`, and fiducials explicitly. Prepared importance models
cache proposal log-probabilities, redshift interpolants, and rate metadata; detector
state (banded `frequencies`, network [`effective_psd`](@ref), observation time) is
passed to inference entry points as flattened arrays.

Inference state is a flat hyperparameter `NamedTuple`. The caller-owned model contract and
Turing integration live in `AstroSGWBInference`; this package provides the reusable physics
and array kernels used to implement that contract.
"""
module AstroSGWB

using CBCDistributions
using Cosmology
import CBCDistributions: single_event_prior
import Cosmology: apply_gw_distance_correction, apply_gw_distance_correction!,
                  cosmology, cosmology_type,
                  gw_em_distance_ratio, hyperparameters,
                  propagation, propagation_type

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

# Types
export canonical_hyperparameters,
       validate_hyperparameters,
       CumulativeIntegral1D,
       GridInterpolator,
       interpolate,
       cdf,
       normalizer,
       RedshiftPrior,
       redshift

# Catalog I/O
export SGWBCatalog,
       load_catalog,
       average_mode

# Detector network (ORF / PSD effective strain PSD and per-bin Gaussian scales)
export Detector,
       PowerSpectralDensity,
       default_detector_data_dir,
       overlap_reduction_function,
       pairwise_overlap_reduction_function,
       effective_psd,
       gaussian_bin_scale,
       gaussian_bin_variance,
       frequency_bin_width

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
       differential_comoving_volume,
       distance_and_volume_grid,
       trapz,
       cumtrapz,
       gw_em_distance_ratio,
       apply_gw_distance_correction,
       apply_gw_distance_correction!,
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
       redshift_integral,
       expected_number_of_events,
       merger_rate_per_sec

# Priors
export OrderedUniformSourceMassPair,
       AlignedSpinChiSimple,
       RedshiftInterpolatedDistribution

# Spectral density
export spectral_density,
       AbstractAverageMode,
       AnalyticInclination,
       CatalogInclination,
       inclination_factor,
       average_mode_config_name,
       average_mode_type,
       SUPPORTED_AVERAGE_MODES,
       inner_product,
       spectral_snr_squared,
       spectral_snr,
       Ωgw

# Time conversions
export JULIAN_YEAR_SEC,
       year_to_second,
       second_to_year

end
