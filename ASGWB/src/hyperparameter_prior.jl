using Distributions
using Distributions: ProductNamedTupleDistribution
using Random

"""
    build_uniform_priors(bounds) -> ProductNamedTupleDistribution

Build the seven-parameter uniform hyperparameter prior as a native
[`Distributions.product_distribution`](@ref) keyed by [`DEFAULT_PARAMETER_ORDER`](@ref).
`bounds` is a dict keyed by parameter name (`"H0"`, `"Omega_m"`, `"chi0"`, `"chin"`,
`"gamma"`, `"kappa"`, `"z_peak"`) carrying `(low, high)` tuples.
"""
function build_uniform_priors(bounds::AbstractDict{
        <:AbstractString, <:Tuple{<:Real, <:Real}})
    return product_distribution((
        H0 = Uniform(Float64(bounds["H0"][1]), Float64(bounds["H0"][2])),
        Ωm = Uniform(Float64(bounds["Omega_m"][1]), Float64(bounds["Omega_m"][2])),
        Ξ₀ = Uniform(Float64(bounds["chi0"][1]), Float64(bounds["chi0"][2])),
        Ξₙ = Uniform(Float64(bounds["chin"][1]), Float64(bounds["chin"][2])),
        γ = Uniform(Float64(bounds["gamma"][1]), Float64(bounds["gamma"][2])),
        κ = Uniform(Float64(bounds["kappa"][1]), Float64(bounds["kappa"][2])),
        zpeak = Uniform(Float64(bounds["z_peak"][1]), Float64(bounds["z_peak"][2]))
    ))
end

"""
    logprior(h::HyperParameters, prior::ProductNamedTupleDistribution) -> Real

Log-prior of `h` under the seven-parameter product distribution. `h` is a flat
`NamedTuple` matching `keys(prior.dists)` (`DEFAULT_PARAMETER_ORDER`).
"""
function logprior(h::HyperParametersNT, prior::ProductNamedTupleDistribution)
    return logpdf(prior, h)
end
