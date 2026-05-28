"""
    ASGWB

Astrophysical stochastic gravitational-wave background modeling: importance
sampling, redshift grids, and likelihoods. MCMC via Turing/AdvancedHMC lives in the
`ASGWBInference` package (see the `ASGWBInference/` directory in the repository).

The inference input artifacts are two files:
- **`model.toml`** — structural model name, cosmology type + parameters,
  modified-gravity Ξ₀/Ξₙ, merger-rate population, and redshift grid settings.
- **`bundle.h5`** ([`WaveformCatalog`](@ref)) — per-sample intrinsic parameters with
  precomputed luminosity distances, and a `(n_freq, n_samples)` per-sample flux matrix
  `|h_+|² + |h_×|²` (before the fiducial `(D_L/D_gw)²` factor).

Use [`load_problem`](@ref) to load both files and build an in-memory
[`ImportanceSamplingProblem`](@ref). Use [`importance_sampling_problem`](@ref) to build
problems directly from in-memory objects (primarily for tests).

Inference state is a flat hyperparameter `NamedTuple` validated against an
[`AbstractASGWBModel`](@ref) contract; the problem carries the structural model and
canonical fiducial hyperparameters directly.
"""
module ASGWB

using CBCDistributions
import CBCDistributions: cosmology, cosmology_type, gravitational_wave_distance

include("types.jl")
include("models/base.jl")
include("models/madau_dickinson.jl")
include("bundle.jl")
include("inference_types.jl")
include("detector/psd.jl")
include("detector/detector.jl")
include("detector/overlap.jl")
include("detector/effective_psd.jl")
include("detector/observation.jl")
include("reconstruction.jl")
include("importance.jl")
include("spectral_density.jl")
include("snr.jl")
include("diagnostics.jl")
include("posterior.jl")
include("io.jl")

# Types
export ImportanceSamplingProblem,
       importance_sampling_problem,
       ProposalData,
       ObservationConfig,
       RedshiftPriorSpec,
       RedshiftPriorFamily,
       MadauDickinson,
       PowerLaw,
       parse_redshift_prior_family,
       AbstractASGWBModel,
       MadauDickinsonModifiedPropagation,
       hyperparameters,
       model_parameters,
       cosmology_type,
       canonical_hyperparameters,
       validate_hyperparameters,
       validate_prior,
       validate_subset,
       ProposalSampleBundle,
       FullBNSSamplesSoA,
       stack_source_masses,
       FULL_BNS_INTRINSIC_ORDER,
       PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR,
       PROPOSAL_SAMPLES_SOURCE_TYPE_BNS,
       CumulativeIntegral1D,
       RedshiftPrior,
       IntrinsicPriorStrategy,
       IntrinsicPrior,
       FullBNS,
       redshift

# Model config
export ModelConfig,
       load_model_config,
       save_model_config,
       model_sha256_of_file,
       model_hyperparameters,
       external_parameter_names,
       external_model_parameter_names,
       redshift_prior_spec,
       reconstruct_proposal_log_prob,
       reconstruct_dgw_fid_sq

# Bundle I/O
export FrequencyGrid,
       frequencies,
       in_band_mask,
       WaveformMetadata,
       WaveformCatalog,
       load_bundle,
       save_bundle,
       verify_model_fingerprint,
       load_problem,
       BUNDLE_COMMAND_ATTR,
       BUNDLE_GIT_REVISION_ATTR

# Detector network (ORF / PSD effective strain PSD; used by `load_problem`)
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
       AbstractCosmology,
       LambdaCDM,
       W0CDM,
       W0WaCDM,
       dark_energy_eos,
       de_density_ratio,
       cosmology_parameters,
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
       inner_product,
       spectral_snr_squared,
       spectral_snr,
       Ωgw,
       evaluate_model_terms

# Diagnostics
export normalized_ess, max_normalized_weight, log_ratio_variance

# Posterior
export loglikelihood,
       logposterior,
       fiducial_hyperparameters,
       fiducial_spectral_density,
       fiducial_redshift_integral

end
