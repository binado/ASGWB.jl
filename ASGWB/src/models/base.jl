using CBCDistributions: AbstractCosmology, cosmology_parameters
using Distributions: ProductNamedTupleDistribution

"""Abstract supertype for ASGWB forward models with explicit hyperparameter contracts."""
abstract type AbstractASGWBModel end

Base.broadcastable(m::AbstractASGWBModel) = Ref(m)

"""
    model_parameters(::Type{<:AbstractASGWBModel}) -> Tuple{Vararg{Symbol}}

Hyperparameter symbols owned by the forward model (excluding cosmology parameters).
"""
function model_parameters end

"""
    external_model_parameter_names(model::AbstractASGWBModel) -> NamedTuple

Maps Julia model parameter symbols to their external string names in `model.toml`.
"""
function external_model_parameter_names end

"""
    model_section_dict(model::AbstractASGWBModel) -> Dict{String, Any}

Serialize the `[model]` table (structural name and cosmology type string).
"""
function model_section_dict end

"""
    redshift_prior_family(model::AbstractASGWBModel) -> RedshiftPriorFamily

Redshift population family used when parsing `[redshift]` from `model.toml`.
"""
function redshift_prior_family end

# --- Generic combinators ---

"""
    hyperparameters(model::AbstractASGWBModel) -> Tuple{Vararg{Symbol}}

Symbols and order used by a model's flat hyperparameter state.
"""
function hyperparameters(m::AbstractASGWBModel)
    return (cosmology_parameters(cosmology_type(m))..., model_parameters(typeof(m))...)
end

"""
    external_parameter_names(model::AbstractASGWBModel) -> NamedTuple

Maps all Julia hyperparameter symbols (cosmology + model-specific) to their external string names.
"""
function external_parameter_names(model::AbstractASGWBModel)
    return (; external_parameter_names(cosmology_type(model))...,
        external_model_parameter_names(model)...)
end

"""
    cosmology(model::AbstractASGWBModel, h::NamedTuple) -> AbstractCosmology

Construct the cosmology for `model` from live hyperparameter state `h`.
"""
function cosmology(m::AbstractASGWBModel, h::NamedTuple)
    return cosmology(cosmology_type(m), h)
end

# --- Cosmology-type ASCII↔Unicode maps ---

external_parameter_names(::Type{LambdaCDM}) = (H0 = "H0", Ωm = "Omega_m")

external_parameter_names(::Type{W0CDM}) = (H0 = "H0", Ωm = "Omega_m", w0 = "w0")

function external_parameter_names(::Type{W0WaCDM})
    return (H0 = "H0", Ωm = "Omega_m", w0 = "w0", wa = "wa")
end

# --- Validators ---

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
