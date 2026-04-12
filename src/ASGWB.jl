"""
    ASGWB

Astrophysical stochastic gravitational-wave background modeling: importance
caches, redshift grids, likelihoods, and sampling (AdvancedHMC and Turing).

Use [`importance_sampling_problem`](@ref) to build problems in memory, or
[`load_cache`](@ref) to read the Julia HDF5 cache format (`format_version` 1, 2, or 3).
For version 2 caches that omit `covariance` / `sgwb_scale`, pass `detectors=` to
[`load_cache`](@ref) so those fields are rebuilt from [`Detector`](@ref) PSDs and ORFs.
Version 3 stores raw per-sample `cached_flux` and may omit `proposal_log_prob` and
`dgw_fid_sq` when fiducial population scalars are present in `hyperparameters`.
Inference state is a nested [`HyperParameters`](@ref); caches carry
[`ProposalFiducialParameters`](@ref) in `fiducial_parameters` (HDF5 group `hyperparameters`).
"""
module ASGWB

include("types.jl")
include("inference_types.jl")
include("detector/psd.jl")
include("detector/detector.jl")
include("detector/overlap.jl")
include("detector/covariance.jl")
include("detector/observation.jl")
include("cosmology.jl")
include("redshift.jl")
include("priors.jl")
include("cache.jl")
include("io.jl")
include("importance.jl")
include("diagnostics.jl")
include("posterior.jl")
include("sampling.jl")
include("turing_model.jl")

# Types
export ImportanceSamplingProblem, ImportanceCache,
    importance_sampling_problem,
    ProposalData, ObservationConfig,
    RedshiftPriorSpec, RedshiftPriorFamily, MadauDickinson, PowerLaw,
    parse_redshift_prior_family,
    HyperParameters,
    CosmologicalParameters,
    ModifiedPropagationParameters,
    PopulationParameters,
    MadauDickinsonParameters,
    PowerLawRedshiftParameters,
    InferencePriors,
    ProposalFiducialParameters,
    ProposalSampleBundle,
    RedshiftOnlySamples,
    FullBNSSamples,
    as_flat_constrained,
    validate_redshift_spec_population,
    RedshiftGridBundle,
    IntrinsicPriorStrategy, RedshiftOnly, FullBNS,
    ASGWBLogDensity, redshift

# IO
export load_cache

# Detector network (ORF / PSD covariance; optional HDF5 format v2)
export Detector, PowerSpectralDensity, default_detector_data_dir,
    overlap_reduction_function, pairwise_overlap_reduction_function,
    covariance_on_grid, gaussian_bin_scale, gaussian_bin_variance,
    frequency_bin_width, build_observation_config

# Cosmology
export E, comoving_distance, luminosity_distance,
    differential_comoving_volume, gravitational_wave_distance

# Redshift & population
export madau_dickinson_source_frame_distribution,
    power_law_source_frame_distribution,
    detector_frame_merger_rate_density,
    build_redshift_grid_bundle,
    log_prob_from_bundle,
    expected_number_of_events

# Priors
export logprior, build_uniform_priors

# Importance sampling
export importance_weights, spectral_density_from_cache,
    evaluate_importance_terms

# Diagnostics
export normalized_ess, max_normalized_weight, log_ratio_variance

# Posterior
export loglikelihood, logposterior

# Sampling (AdvancedHMC)
export DEFAULT_PARAMETER_ORDER,
    build_prior_distribution,
    unconstrained_initial_point, constrained_parameters,
    ad_logdensity, finite_difference_logdensity_and_gradient,
    sample_with_advancedhmc

# Turing
export build_turing_model, sample_with_turing

end
