module InferenceImpl

using AstroSGWB
using AstroSGWB:
                 AbstractAverageMode,
                 AnalyticInclination,
                 CatalogInclination,
                 inner_product,
                 frequency_bin_width,
                 gaussian_bin_scale,
                 year_to_second
using Distributions: Distributions, MvNormal
using LinearAlgebra: Diagonal
using Random: Random
using Turing

include("forward.jl")
include("diagnostics.jl")
include("amplitude.jl")
include("reconstruction.jl")
include("turing_model.jl")

export forward_model,
       astrosgwb_importance_turing_model,
       astrosgwb_amplitude_marginalized_turing_model,
       AmplitudeConditional,
       quadrature_grid,
       log_normalizer,
       effective_nodes,
       reconstruct_amplitude,
       AbstractAverageMode,
       AnalyticInclination,
       CatalogInclination

end
