module InferenceImpl

using AstroSGWB
using AstroSGWB:
                 AbstractAverageMode,
                 AnalyticInclination,
                 CatalogInclination,
                 spectral_snr,
                 frequency_bin_width,
                 gaussian_bin_scale,
                 year_to_second
using Distributions: MvNormal
using LinearAlgebra: Diagonal
using Turing

include("forward.jl")
include("diagnostics.jl")
include("turing_model.jl")

export forward_model,
       astrosgwb_importance_turing_model,
       AbstractAverageMode,
       AnalyticInclination,
       CatalogInclination

end
