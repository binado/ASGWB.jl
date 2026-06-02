module InferenceImpl

using ASGWB
using ASGWB:
             ImportanceSamplingProblem,
             ModelContext,
             PopulationModel,
             AbstractCosmology,
             SUPPORTED_COSMOLOGIES,
             loglikelihood,
             merger_rate,
             compute_importance_weights,
             spectral_density,
             canonical_hyperparameters,
             full_hyperparameters,
             validate_subset,
             normalized_ess,
             spectral_snr_squared,
             frequency_bin_width
using Distributions: MvNormal, ProductNamedTupleDistribution, logpdf
using LinearAlgebra: Diagonal
using Turing

include("turing_model.jl")

export build_turing_model,
       condition_turing_model,
       logposterior,
       validate_hyperprior

end
