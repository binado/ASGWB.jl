using CBCDistributions: AbstractCosmology, ParametrizedModel, PhysicalModel,
                        PopulationModel, PureModel, MadauDickinsonSourceFrameModel,
                        ModifiedPropagation, hyperparameters
using Distributions: ProductNamedTupleDistribution

const AbstractASGWBModel = ParametrizedModel

cosmology_type(model::PhysicalModel) = model.cosmology_type

external_parameter_names(::Type{LambdaCDM}) = (H0 = "H0", Ωm = "Omega_m")
external_parameter_names(::Type{W0CDM}) = (H0 = "H0", Ωm = "Omega_m", w0 = "w0")
function external_parameter_names(::Type{W0WaCDM})
    (H0 = "H0", Ωm = "Omega_m", w0 = "w0", wa = "wa")
end
function external_parameter_names(::Type{<:ModifiedPropagation{C}}) where {C <:
                                                                           AbstractCosmology}
    return (; external_parameter_names(C)..., Ξ₀ = "Xi_0", Ξₙ = "Xi_n")
end

function external_parameter_names(::MadauDickinsonSourceFrameModel)
    (γ = "gamma", κ = "kappa", zpeak = "z_peak")
end
external_parameter_names(::PureModel) = NamedTuple()

function external_parameter_names(model::PopulationModel)
    parts = NamedTuple()
    for component in values(model.components)
        parts = merge(parts, external_parameter_names(component))
    end
    return parts
end

function external_parameter_names(model::PhysicalModel)
    (; external_parameter_names(model.cosmology_type)...,
        external_parameter_names(model.population)...)
end

external_model_parameter_names(model::ParametrizedModel) = external_parameter_names(model)

function model_parameters(::Type{<:ParametrizedModel})
    return ()
end

function _check_unique_hyperparameters(model::ParametrizedModel)
    order = hyperparameters(model)
    isempty(order) &&
        throw(ArgumentError("$(typeof(model)) must define at least one hyperparameter"))
    length(unique(order)) == length(order) ||
        throw(ArgumentError("$(typeof(model)) defines duplicate hyperparameters: $(order)"))
    return order
end

function validate_subset(
        subset::Tuple{Vararg{Symbol}},
        order::Union{Tuple{Vararg{Symbol}}, Base.KeySet, AbstractVector{Symbol}}
)
    for s in subset
        s in order ||
            throw(ArgumentError("subset contains $(repr(s)); expected symbols from $(Tuple(order))"))
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

function validate_subset(subset, model::ParametrizedModel)
    validate_subset(subset, _check_unique_hyperparameters(model))
end

function validate_subset(subset, prior::ProductNamedTupleDistribution)
    validate_subset(subset, keys(prior.dists))
end

function validate_hyperparameters(
        model::ParametrizedModel,
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters"
)
    order = _check_unique_hyperparameters(model)
    keys(Λ) == order || throw(
        ArgumentError("$(context) must exactly match $(typeof(model)); expected $(order), got $(keys(Λ))"),
    )
    return nothing
end

function canonical_hyperparameters(
        model::ParametrizedModel,
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters",
        eltype = Float64
)
    order = _check_unique_hyperparameters(model)
    Set(keys(Λ)) == Set(order) || throw(
        ArgumentError("$(context) must exactly match $(typeof(model)); expected $(order), got $(keys(Λ))"),
    )
    eltype === nothing && return (; (k => Λ[k] for k in order)...)
    return (; (k => eltype(Λ[k]) for k in order)...)
end

function require_redshift_population(model::PhysicalModel)
    :redshift in keys(model.population.components) ||
        throw(ArgumentError("ASGWB physical models require a :redshift population component"))
    return nothing
end
