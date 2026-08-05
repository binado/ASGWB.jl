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
using Distributions: MvNormal
using LinearAlgebra: Diagonal
using Turing

include("forward.jl")
include("turing_model.jl")

export forward_model,
       build_turing_model,
       AbstractAverageMode,
       AnalyticInclination,
       CatalogInclination

end
