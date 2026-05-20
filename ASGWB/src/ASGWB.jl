"""
    ASGWB

Astrophysical stochastic gravitational-wave background modeling: importance
caches, redshift grids, and likelihoods. MCMC via Turing/AdvancedHMC lives in the
`ASGWBInference` package (see the `ASGWBInference/` directory in the repository).

Use [`importance_sampling_problem`](@ref) to build problems in memory, or
[`load_cache`](@ref) to read the HDF5 importance cache. Caches record provenance via root
attributes [`IMPORTANCE_CACHE_COMMAND_ATTR`](@ref) and [`IMPORTANCE_CACHE_GIT_REVISION_ATTR`](@ref)
(`command` and `git_revision`). Pass a vector of at least two [`Detector`](@ref) values as the
second argument so `effective_psd` and `sgwb_scale` are built from tabulated PSDs and ORFs (those
datasets must not appear in the file). Two-dimensional datasets `cached_flux` and
`proposal_intrinsic_vector` use HDF5 extent `(n_columns, n_samples)` and are normalized to
`(n_samples, n_columns)` on load. Per-sample flux is stored as `cached_flux` (before the
fiducial ``(D_L/D_{gw})^2`` factor). Datasets `proposal_log_prob` and `dgw_fid_sq` may be omitted
and are then reconstructed. Population scalars may live in `hyperparameters` and/or
`redshift_prior_spec` (duplicate keys must agree). An HDF5 `fiducial_spectral_density` dataset, if
present, is ignored on load; [`load_cache`](@ref) always fills the observation using
[`fiducial_spectral_density`](@ref) so the default likelihood data match the current Julia pipeline.
Caches may omit
`redshift_integral_fiducial`; it is then set from [`fiducial_redshift_integral`](@ref).
Inference state is a flat [`HyperParameters`](@ref) `NamedTuple`; caches carry
[`ProposalFiducialParameters`](@ref) in `fiducial_parameters` (HDF5 group `hyperparameters`).
"""
module ASGWB

using CBCDistributions

include("types.jl")
include("inference_types.jl")
include("detector/psd.jl")
include("detector/detector.jl")
include("detector/overlap.jl")
include("detector/effective_psd.jl")
include("detector/observation.jl")
include("cache.jl")
include("importance.jl")
include("spectral_density.jl")
include("snr.jl")
include("diagnostics.jl")
include("posterior.jl")
include("parity_test_cache.jl")
include("io.jl")

# Types
export ImportanceSamplingProblem,
       ImportanceCache,
       importance_sampling_problem,
       ProposalData,
       ObservationConfig,
       RedshiftPriorSpec,
       RedshiftPriorFamily,
       MadauDickinson,
       PowerLaw,
       parse_redshift_prior_family,
       HyperParameters,
       ProposalFiducialParameters,
       ProposalSampleBundle,
       FullBNSSamplesSoA,
       stack_source_masses,
       FULL_BNS_INTRINSIC_ORDER,
       PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR,
       PROPOSAL_SAMPLES_SOURCE_TYPE_BNS,
       CumulativeIntegral1D,
       Cosmology,
       CosmologyCache,
       RedshiftPrior,
       IntrinsicPriorStrategy,
       IntrinsicPrior,
       FullBNS,
       redshift

# IO
export parity_cache_path,
       resolve_parity_cache_path,
       write_parity_cache_h5,
       load_cache,
       reconstruct_proposal_log_prob,
       reconstruct_dgw_fid_sq,
       IMPORTANCE_CACHE_COMMAND_ATTR,
       IMPORTANCE_CACHE_GIT_REVISION_ATTR

# Detector network (ORF / PSD effective strain PSD; used by `load_cache`)
export Detector,
       PowerSpectralDensity,
       default_detector_data_dir,
       overlap_reduction_function,
       pairwise_overlap_reduction_function,
       effective_psd,
       gaussian_bin_scale,
       gaussian_bin_variance,
       frequency_bin_width,
       build_observation_config

# Cosmology
export E,
       Cosmology,
       CosmologyCache,
       comoving_distance,
       luminosity_distance,
       differential_comoving_volume,
       gravitational_wave_distance

# Redshift & population
export madau_dickinson_source_frame_distribution,
       power_law_source_frame_distribution,
       detector_frame_merger_rate_density,
       build_redshift_prior,
       cosmology_and_redshift_prior,
       redshift_log_prob,
       redshift_log_prob_samples,
       redshift_log_prob_samples!,
       redshift_logpdf_eltype,
       redshift_integral,
       expected_number_of_events,
       merger_rate_per_sec

# Priors
export OrderedUniformSourceMassPair,
       AlignedSpinChiSimple,
       RedshiftInterpolatedDistribution,
       intrinsic_prior,
       validate_batch

# Importance sampling
export importance_weights,
       compute_importance_weights,
       spectral_density,
       spectral_snr_squared,
       spectral_snr,
       Ωgw,
       evaluate_importance_terms

# Diagnostics
export normalized_ess, max_normalized_weight, log_ratio_variance

# Posterior
export loglikelihood,
       logposterior,
       fiducial_hyperparameters,
       fiducial_spectral_density,
       fiducial_redshift_integral

# Hyperparameter ordering (used with product priors / Bijectors in ASGWBInference)
export DEFAULT_PARAMETER_ORDER

end
