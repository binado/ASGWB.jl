using CBCDistributions: AbstractCosmology, cosmology_parameters
import CBCDistributions: cosmology
using Distributions: ProductNamedTupleDistribution

"""Abstract supertype for ASGWB forward models with explicit hyperparameter contracts."""
abstract type AbstractASGWBModel end

"""
Madau-Dickinson population with modified gravitational-wave propagation.

Type parameter `C <: AbstractCosmology` selects the cosmology model.
`MadauDickinsonModifiedPropagation()` defaults to `LambdaCDM`.
"""
struct MadauDickinsonModifiedPropagation{C <: AbstractCosmology} <: AbstractASGWBModel end

MadauDickinsonModifiedPropagation() = MadauDickinsonModifiedPropagation{LambdaCDM}()

"""
    hyperparameters(model::AbstractASGWBModel) -> Tuple{Vararg{Symbol}}

Symbols and order used by a model's flat hyperparameter state.
"""
function hyperparameters end

"""
    model_parameters(::Type{<:MadauDickinsonModifiedPropagation}) -> Tuple{Vararg{Symbol}}

Hyperparameter symbols owned by the Madau–Dickinson modified-propagation forward model
(excluding cosmology parameters).
"""
model_parameters(::Type{<:MadauDickinsonModifiedPropagation}) = (:Ξ₀, :Ξₙ, :γ, :κ, :zpeak)

function hyperparameters(::Type{MadauDickinsonModifiedPropagation{C}}) where {C <:
                                                                              AbstractCosmology}
    return (
        cosmology_parameters(C)...,
        model_parameters(MadauDickinsonModifiedPropagation{C})...
    )
end

hyperparameters(m::MadauDickinsonModifiedPropagation) = hyperparameters(typeof(m))

"""
    cosmology(model::MadauDickinsonModifiedPropagation{C}, h::NamedTuple) -> AbstractCosmology

Construct the cosmology for `model` from live hyperparameter state `h` (delegates to [`CBCDistributions.cosmology`](@ref)).
"""
function cosmology(m::MadauDickinsonModifiedPropagation{C}, h::NamedTuple) where {C <:
                                                                                  AbstractCosmology}
    cosmology(C, h)
end

"""
    cosmology_type(model::MadauDickinsonModifiedPropagation{C}) -> Type{C}

Return the cosmology subtype baked into the forward model's type parameter.
"""
cosmology_type(::MadauDickinsonModifiedPropagation{C}) where {C <: AbstractCosmology} = C

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
