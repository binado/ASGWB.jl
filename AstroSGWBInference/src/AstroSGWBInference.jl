module AstroSGWBInference

include("InferenceImpl.jl")
using .InferenceImpl:
                      hyperparameters,
                      merger_rate_and_log_weights,
                      fiducial_spectral_density,
                      build_turing_model,
                      condition_turing_model,
                      loglikelihood,
                      logposterior,
                      AbstractAverageMode,
                      AnalyticInclination,
                      CatalogInclination

export hyperparameters,
       merger_rate_and_log_weights,
       fiducial_spectral_density,
       build_turing_model,
       condition_turing_model,
       loglikelihood,
       logposterior,
       AbstractAverageMode,
       AnalyticInclination,
       CatalogInclination,
       atomic_save_chain,
       MCMCConfig,
       SamplerConfig,
       load_config,
       save_config,
       validate_fiducials

include("chain_io.jl")
include("config.jl")
include("cli/stack_partial_chains.jl")
using .ChainIO: atomic_save_chain
using .Config: MCMCConfig, SamplerConfig, load_config, save_config, validate_fiducials

end
