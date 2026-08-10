using Distributions

"""
    PopulationModel

Abstract supertype for caller-defined population models.  Concrete subtypes
must implement the two-method contract:

- `hyperparameters(pop) -> NTuple{N, Symbol}` — ordered population parameter names.
- `single_event_prior(pop, cosmology, Λ; z_grid) -> ProductNamedTupleDistribution`
  — per-event distribution conditioned on the cosmology and hyperparameters `Λ`.
  Build the redshift component with `redshift_prior(sf_model, cosmology, Λ; z_grid)`.

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
    single_event_prior(pop, cosmology, Λ; z_grid) -> ProductNamedTupleDistribution

Per-event distribution over intrinsic parameters for a cosmology and hyperparameter
state `Λ`. Implement on concrete `PopulationModel` subtypes, threading `z_grid` into
`redshift_prior` when the population includes redshift.
"""
function single_event_prior end

"""
    full_hyperparameters(C, P, pop) -> NTuple{N,Symbol}

Concatenation of cosmology, propagation, and population hyperparameter symbols, in
the order used for the flat HMC/Turing parameter vector: `(cosmo…, Ξ₀, Ξₙ, pop…)`.
"""
function full_hyperparameters(
        ::Type{C}, ::Type{P},
        pop::PopulationModel
) where {C <: AbstractCosmology, P <: AbstractPropagation}
    return (Cosmology.hyperparameters(C)...,
        Cosmology.propagation_hyperparameters(P)...,
        hyperparameters(pop)...)
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
