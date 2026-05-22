using Distributions: ProductNamedTupleDistribution

"""Abstract supertype for ASGWB forward models with explicit hyperparameter contracts."""
abstract type AbstractASGWBModel end

"""Madau-Dickinson population with modified gravitational-wave propagation."""
struct MadauDickinsonModifiedPropagation <: AbstractASGWBModel end

"""
    hyperparameters(model::AbstractASGWBModel) -> Tuple{Vararg{Symbol}}

Symbols and order used by a model's flat hyperparameter state.
"""
function hyperparameters end

hyperparameters(::MadauDickinsonModifiedPropagation) = (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)

"""
    hyperparameter_order(prior::ProductNamedTupleDistribution)

Symbols and order used by `Bijectors.link` / HMC unconstrained vectors (`keys(prior.dists)`).
Compatibility helper for prior order. Prefer [`hyperparameters`](@ref) on an
[`AbstractASGWBModel`](@ref) when validating model state.
"""
hyperparameter_order(prior::ProductNamedTupleDistribution) = keys(prior.dists)

hyperparameter_order(model::AbstractASGWBModel) = hyperparameters(model)

function _check_unique_hyperparameters(model::AbstractASGWBModel)
    order = hyperparameters(model)
    isempty(order) && throw(
        ArgumentError("$(typeof(model)) must define at least one hyperparameter"),
    )
    length(unique(order)) == length(order) || throw(
        ArgumentError("$(typeof(model)) defines duplicate hyperparameters: $(order)"),
    )
    return order
end

"""
    validate_subset(subset::Union{Tuple{Vararg{Symbol}}, NamedTuple}, order) -> subset

Validate that `subset` (either a tuple of symbols or a NamedTuple) contains only
unique symbols that are a subset of `order`. Allows empty subsets. Throws `ArgumentError`
on duplicates or unknown symbols.
"""
function validate_subset(
        subset::Tuple{Vararg{Symbol}},
        order::Union{Tuple{Vararg{Symbol}}, Base.KeySet, AbstractVector{Symbol}}
)
    for s in subset
        s in order || throw(
            ArgumentError(
            "subset contains $(repr(s)); expected symbols from $(Tuple(order))",
        ),
        )
    end
    length(unique(subset)) == length(subset) ||
        throw(ArgumentError("subset must not repeat symbols"))
    return subset
end

function validate_subset(
        subset::NamedTuple,
        order::Union{Tuple{Vararg{Symbol}}, Base.KeySet, AbstractVector{Symbol}}
)
    validate_subset(keys(subset), order)
    return subset
end

"""
    validate_subset(subset, model::AbstractASGWBModel) -> subset
"""
function validate_subset(subset, model::AbstractASGWBModel)
    validate_subset(subset, _check_unique_hyperparameters(model))
end

"""
    validate_subset(subset, prior::ProductNamedTupleDistribution) -> subset
"""
function validate_subset(subset, prior::ProductNamedTupleDistribution)
    validate_subset(subset, keys(prior.dists))
end

"""Compatibility alias for [`validate_subset`](@ref)."""
validate_subset!(args...; kwargs...) = validate_subset(args...; kwargs...)

"""
    validate_hyperparameters(model, Λ; context="hyperparameters")

Require `Λ` to contain exactly the model hyperparameters.
"""
function validate_hyperparameters(
        model::AbstractASGWBModel,
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters"
)
    validate_subset(Λ, model)
    order = hyperparameters(model)
    if length(keys(Λ)) != length(order)
        missing = Tuple(s for s in order if s ∉ keys(Λ))
        throw(ArgumentError("$(context) must match $(typeof(model)); missing $(missing)"))
    end
    return nothing
end

"""
    canonical_hyperparameters(model, Λ; context="hyperparameters", eltype=Float64) -> NamedTuple

Validate a hyperparameter tuple against `model`, reorder it to the model's canonical
hyperparameter order, and convert each value to `eltype`.
"""
function canonical_hyperparameters(
        model::AbstractASGWBModel,
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters",
        eltype::Type = Float64
)
    validate_hyperparameters(model, Λ; context = context)
    return (; (k => eltype(Λ[k]) for k in hyperparameters(model))...)
end

"""
    validate_prior(model, prior)

Require a product prior's named sites and order to match the model hyperparameters.
"""
function validate_prior(
        model::AbstractASGWBModel,
        prior::ProductNamedTupleDistribution
)
    order = _check_unique_hyperparameters(model)
    prior_order = keys(prior.dists)
    prior_order == order || throw(
        ArgumentError(
        "prior hyperparameters must match $(typeof(model)); expected $(order), got $(prior_order)",
    ),
    )
    return nothing
end

"""
    validate_sample_only(sample_only, model::AbstractASGWBModel)

Validate `sample_only` against [`hyperparameters`](@ref). Pass `nothing` to sample all
hyperparameters. Throws `ArgumentError` on empty, duplicate, or unknown symbols.
"""
function validate_sample_only(
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}},
        model::AbstractASGWBModel
)
    sample_only === nothing && return nothing
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
    ),
    )
    validate_subset(sample_only, model)
    return nothing
end

function validate_sample_only(
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}},
        prior::ProductNamedTupleDistribution
)
    sample_only === nothing && return nothing
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
    ),
    )
    validate_subset(sample_only, prior)
    return nothing
end

"""Compatibility alias for [`validate_sample_only`](@ref)."""
validate_sample_only!(args...; kwargs...) = validate_sample_only(args...; kwargs...)
