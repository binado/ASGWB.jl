"""
    ASGWB

Astrophysical stochastic gravitational-wave background modeling: importance
caches, redshift grids, likelihoods, and sampling (AdvancedHMC and Turing).

Use [`importance_sampling_problem`](@ref) to build problems in memory, or
[`load_cache`](@ref) to read the Julia HDF5 cache format.
"""
module ASGWB

include("types.jl")
include("io.jl")
include("cosmology.jl")
include("redshift.jl")
include("priors.jl")
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
    RedshiftGridBundle,
    IntrinsicPriorStrategy, RedshiftOnly, FullBNS,
    ASGWBLogDensity, redshift

# IO
export load_cache

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
