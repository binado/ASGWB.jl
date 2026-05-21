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
    validate_hyperparameters(model, Λ; context="hyperparameters")

Require `Λ` to contain exactly the model hyperparameters.
"""
function validate_hyperparameters(
        model::AbstractASGWBModel,
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters"
)
    order = _check_unique_hyperparameters(model)
    names = keys(Λ)
    Set(names) == Set(order) && return nothing

    missing = Tuple(s for s in order if s ∉ names)
    extra = Tuple(s for s in names if s ∉ order)
    parts = String[]
    isempty(missing) || push!(parts, "missing $(missing)")
    isempty(extra) || push!(parts, "extra $(extra)")
    throw(ArgumentError("$(context) must match $(typeof(model)); " * join(parts, "; ")))
end

"""
    float_hyperparameters(model, Λ; context="hyperparameters") -> NamedTuple

Validate and convert a model hyperparameter tuple to `Float64` in model order.
"""
function float_hyperparameters(
        model::AbstractASGWBModel,
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters"
)
    validate_hyperparameters(model, Λ; context = context)
    return (; (k => Float64(Λ[k]) for k in hyperparameters(model))...)
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
    validate_sample_only!(sample_only, model::AbstractASGWBModel)

Validate `sample_only` against [`hyperparameters`](@ref). Pass `nothing` to sample all
hyperparameters. Throws `ArgumentError` on empty, duplicate, or unknown symbols.
"""
function validate_sample_only!(
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}},
        model::AbstractASGWBModel
)
    sample_only === nothing && return nothing
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
    ),
    )
    order = _check_unique_hyperparameters(model)
    for s in sample_only
        s in order || throw(
            ArgumentError(
            "sample_only contains $(repr(s)); expected symbols from $(Tuple(order))",
        ),
        )
    end
    length(unique(sample_only)) == length(sample_only) ||
        throw(ArgumentError("sample_only must not repeat symbols"))
    return nothing
end

function validate_sample_only!(
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}},
        prior::ProductNamedTupleDistribution
)
    sample_only === nothing && return nothing
    order = keys(prior.dists)
    model = MadauDickinsonModifiedPropagation()
    order == hyperparameters(model) && return validate_sample_only!(sample_only, model)
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
    ),
    )
    for s in sample_only
        s in order || throw(
            ArgumentError(
            "sample_only contains $(repr(s)); expected symbols from $(Tuple(order))",
        ),
        )
    end
    length(unique(sample_only)) == length(sample_only) ||
        throw(ArgumentError("sample_only must not repeat symbols"))
    return nothing
end

"""
    coerce_hyperparameters(; H0, Ωm, Ξ₀=1.0, Ξₙ=0.0, γ, κ, zpeak) -> NamedTuple

Legacy wrapper for [`float_hyperparameters`](@ref) on
[`MadauDickinsonModifiedPropagation`](@ref).
Inner likelihood paths accept any `NamedTuple` (including `ForwardDiff.Dual` fields during AD).
"""
function coerce_hyperparameters(;
        H0::Real,
        Ωm::Real,
        Ξ₀::Real = 1.0,
        Ξₙ::Real = 0.0,
        γ::Real,
        κ::Real,
        zpeak::Real
)
    return float_hyperparameters(
        MadauDickinsonModifiedPropagation(),
        (; H0, Ωm, Ξ₀, Ξₙ, γ, κ, zpeak);
        context = "Madau-Dickinson modified-propagation hyperparameters"
    )
end

"""
    coerce_hyperparameters(nt::NamedTuple) -> NamedTuple

Build a `Float64` hyperparameter `NamedTuple` from any tuple with at least
`:H0, :Ωm, :γ, :κ, :zpeak`. `Ξ₀` / `Ξₙ` default to `1.0` / `0.0` when absent.
"""
function coerce_hyperparameters(nt::NamedTuple)
    return coerce_hyperparameters(;
        H0 = nt.H0,
        Ωm = nt.Ωm,
        Ξ₀ = haskey(nt, :Ξ₀) ? nt.Ξ₀ : 1.0,
        Ξₙ = haskey(nt, :Ξₙ) ? nt.Ξₙ : 0.0,
        γ = nt.γ,
        κ = nt.κ,
        zpeak = nt.zpeak
    )
end
