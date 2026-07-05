module InferenceImpl

using AstroSGWB
using AstroSGWB:
                 ObservationContext,
                 normalized_ess,
                 spectral_snr_squared,
                 frequency_bin_width,
                 year_to_second
using Distributions: MvNormal, ProductNamedTupleDistribution, logpdf
using LinearAlgebra: Diagonal
using Turing

include("models/base.jl")
include("likelihood.jl")
include("turing_model.jl")

export hyperparameters,
       merger_rate_and_log_weights,
       fiducial_spectral_density,
       build_turing_model,
       condition_turing_model,
       loglikelihood,
       logposterior

end
