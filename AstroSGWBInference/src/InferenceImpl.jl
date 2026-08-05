module InferenceImpl

using AstroSGWB
using AstroSGWB:
                 AbstractAverageMode,
                 AnalyticInclination,
                 CatalogInclination,
                 ObservationContext,
                 normalized_ess,
                 spectral_snr_squared,
                 frequency_bin_width,
                 year_to_second
using Distributions: MvNormal, logpdf
using LinearAlgebra: Diagonal
using Turing

include("models/base.jl")
include("likelihood.jl")
include("turing_model.jl")

export fiducial_spectral_density,
       build_turing_model,
       condition_turing_model,
       loglikelihood,
       logposterior,
       AbstractAverageMode,
       AnalyticInclination,
       CatalogInclination

end
