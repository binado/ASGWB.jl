module InferenceImpl

using GWBackground
using GWBackground:
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
using Trapezoid: trapz, cumtrapz
using Turing

include("forward.jl")
include("diagnostics.jl")
include("amplitude.jl")
include("reconstruction.jl")
include("turing_model.jl")

export forward_model,
       gwbackground_importance_turing_model,
       gwbackground_amplitude_marginalized_turing_model,
       AmplitudeConditional,
       quadrature_grid,
       log_normalizer,
       effective_nodes,
       reconstruct_amplitude,
       AbstractAverageMode,
       AnalyticInclination,
       CatalogInclination

end
