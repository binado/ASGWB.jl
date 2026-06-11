using Distributions
using Distributions: ProductNamedTupleDistribution

"""
    PopulationModel

Abstract supertype for caller-defined population models.  Concrete subtypes
must implement the two-method contract:

- `hyperparameters(pop) -> NTuple{N, Symbol}` — ordered population parameter names.
- `single_event_prior(pop, cache::CosmologyCache, Λ) -> ProductNamedTupleDistribution`
  — per-event distribution conditioned on the cosmology carried by `cache` and
  hyperparameters `Λ`. Build the redshift component with
  `redshift_prior(sf_model, cache, Λ)` so the cache is reused. A generic
  `single_event_prior(pop, cosmo::AbstractCosmology, Λ; z_grid)` adapter is provided
  for callers that only hold a bare cosmology.

Hyperparameter priors are caller-defined (e.g. `product_distribution(...)` in
notebooks or tests); they are not part of this package API.
"""
abstract type PopulationModel end

Base.broadcastable(m::PopulationModel) = Ref(m)

"""
    hyperparameters(pop::PopulationModel) -> NTuple{N,Symbol}

Ordered tuple of hyperparameter symbols owned by `pop`.  Implement on concrete
subtypes; do not overlap with the cosmology symbols.
"""
function hyperparameters end

"""
    single_event_prior(pop, cache::CosmologyCache, Λ) -> ProductNamedTupleDistribution

Per-event distribution over intrinsic parameters for the cosmology carried by
`cache` and hyperparameter state `Λ`.  Implement on concrete `PopulationModel`
subtypes, threading `cache` into `redshift_prior` so its cumulative cosmology
integral is reused by the importance-weight path rather than rebuilt.
"""
function single_event_prior end

"""
    single_event_prior(pop, cosmo::AbstractCosmology, Λ; z_grid) -> ProductNamedTupleDistribution

Generic adapter for callers that hold a bare cosmology (the oracle and
fiducial-reconstruction paths). Builds a [`CosmologyCache`](@ref) on `z_grid`
(default [`DEFAULT_Z_GRID`](@ref)) and dispatches to the population's cache method.
The hot path builds the cache once and calls the cache method directly.
"""
function single_event_prior(
        pop::PopulationModel,
        cosmo::AbstractCosmology,
        Λ::NamedTuple;
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
)
    return single_event_prior(pop, CosmologyCache(cosmo, z_grid), Λ)
end

"""
    full_hyperparameters(C, pop) -> NTuple{N,Symbol}

Concatenation of cosmology and population hyperparameter symbols, in the order
used for the flat HMC/Turing parameter vector.
"""
function full_hyperparameters(::Type{C}, pop::PopulationModel) where {C <:
                                                                      AbstractCosmology}
    return (hyperparameters(C)..., hyperparameters(pop)...)
end

"""
    validate_hyperparameters(order, Λ; context) -> nothing

Assert that `keys(Λ) == order` exactly (same symbols, same order).
"""
function validate_hyperparameters(
        order::Tuple{Vararg{Symbol}},
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters"
)
    keys(Λ) == order || throw(
        ArgumentError("$(context) must match order $(order), got $(keys(Λ))"),
    )
    return nothing
end

"""
    canonical_hyperparameters(order, Λ; context, eltype) -> NamedTuple

Re-key `Λ` into the order given by `order`, converting values to `eltype`.
`Λ` may have its keys in any order as long as the set matches `order` exactly.
Pass `eltype = nothing` to preserve original value types.
"""
function canonical_hyperparameters(
        order::Tuple{Vararg{Symbol}},
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters",
        eltype = Float64
)
    Set(keys(Λ)) == Set(order) || throw(
        ArgumentError(
        "$(context) must exactly match $(order), got $(keys(Λ))"),
    )
    eltype === nothing && return (; (k => Λ[k] for k in order)...)
    return (; (k => eltype(Λ[k]) for k in order)...)
end

# ---------------------------------------------------------------------------
# Batched log-pdf helpers (consumed by likelihood and importance-sampling paths)
# ---------------------------------------------------------------------------

function _component_batch_length(d::Distribution, samples::NamedTuple, key)
    haskey(samples, key) ||
        throw(ArgumentError("samples are missing population prior field $(repr(key))"))
    return _component_batch_length(d, samples[key], key)
end

function _component_batch_length(d::UnivariateDistribution, field::AbstractVector, key)
    return length(field)
end

function _component_batch_length(d::MultivariateDistribution, field::AbstractMatrix, key)
    expected = length(d)
    size(field, 1) == expected ||
        throw(
            ArgumentError(
            "population prior field $(repr(key)) must have $expected rows, got $(size(field, 1))",
        ),
        )
    return size(field, 2)
end

function _component_batch_length(d, field, key)
    throw(
        ArgumentError(
        "unsupported batch layout for population prior field $(repr(key)) and distribution $(typeof(d))",
    ),
    )
end

function _batched_output_eltype(dists)
    isempty(dists) && return Float64
    return promote_type(map(eltype, values(dists))...)
end

function _add_component_logpdf!(
        out::AbstractVector,
        d::UnivariateDistribution,
        field::AbstractVector
)
    @inbounds for i in eachindex(out, field)
        out[i] += logpdf(d, field[i])
    end
    return out
end

function _add_component_logpdf!(
        out::AbstractVector,
        d::MultivariateDistribution,
        field::AbstractMatrix
)
    values = logpdf(d, field)
    length(values) == length(out) ||
        throw(ArgumentError("component logpdf length must match output length"))
    @inbounds for i in eachindex(out, values)
        out[i] += values[i]
    end
    return out
end

function _add_component_logpdf!(
        out::AbstractVector,
        d::OrderedUniformSourceMassPair,
        field::AbstractMatrix
)
    size(field, 1) == 2 ||
        throw(ArgumentError("ordered source-mass batch must have two rows"))
    @inbounds for i in eachindex(out)
        out[i] += logpdf(d, (field[1, i], field[2, i]))
    end
    return out
end

"""
    batched_logpdf(d::ProductNamedTupleDistribution, samples::NamedTuple) -> Vector

Per-sample log-density of `d` evaluated against a struct-of-arrays `samples`.
Each field of `d.dists` is matched to the same field in `samples`.
"""
function batched_logpdf(d::ProductNamedTupleDistribution, samples::NamedTuple)
    first_key = first(keys(d.dists))
    n = _component_batch_length(d.dists[first_key], samples, first_key)
    T = _batched_output_eltype(d.dists)
    out = zeros(T, n)
    for key in keys(d.dists)
        n_key = _component_batch_length(d.dists[key], samples, key)
        n_key == n ||
            throw(ArgumentError("population prior sample fields must have matching lengths"))
        _add_component_logpdf!(out, d.dists[key], samples[key])
    end
    return out
end

"""
    batched_logpdf(d::Distribution, samples) -> logpdf(d, samples)

Scalar fallback for non-batched distributions.
"""
function batched_logpdf(d::Distribution, samples)
    return logpdf(d, samples)
end
