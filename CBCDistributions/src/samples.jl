"""
    IntrinsicPriorStrategy

Abstract supertype for proposal-sample intrinsic-prior strategies. Concrete subtypes
(currently [`FullBNS`](@ref)) are used as dispatch tags by [`intrinsic_prior`](@ref).
"""
abstract type IntrinsicPriorStrategy end

"""Full binary neutron star intrinsic variables in proposal samples."""
struct FullBNS <: IntrinsicPriorStrategy end

"""
    stack_source_masses(mass_1_source, mass_2_source) -> Matrix{Float64}

Pack two same-length mass vectors into a `2 × n` matrix (row 1 = `mass_1_source`,
row 2 = `mass_2_source`), the layout the full-BNS intrinsic prior expects under the
`mass` field of a proposal-sample `NamedTuple`.
"""
function stack_source_masses(
        mass_1_source::AbstractVector{<:Real},
        mass_2_source::AbstractVector{<:Real}
)::Matrix{Float64}
    n = length(mass_1_source)
    length(mass_2_source) == n ||
        throw(ArgumentError("mass_1_source and mass_2_source must have matching lengths"))
    return permutedims(
        hcat(collect(Float64, mass_1_source), collect(Float64, mass_2_source)),
    )
end

"""Canonical ordering of full-BNS intrinsic columns on disk."""
const FULL_BNS_INTRINSIC_ORDER = [
    "mass_1_source", "mass_2_source", "redshift", "chi_1", "chi_2", "lambda_1", "lambda_2"]
