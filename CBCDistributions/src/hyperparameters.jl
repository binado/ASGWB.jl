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
