"""
    IntrinsicPriorStrategy

Abstract supertype for proposal-sample intrinsic-prior strategies. Concrete subtypes
(currently [`FullBNS`](@ref)) are used as dispatch tags by [`intrinsic_prior`](@ref) and
[`fixed_intrinsic_log_prob`](@ref).
"""
abstract type IntrinsicPriorStrategy end

"""Full binary neutron star intrinsic variables in proposal samples."""
struct FullBNS <: IntrinsicPriorStrategy end

"""
    FullBNSSamplesSoA

Struct-of-arrays proposal-sample container matching the NamedTuple returned by
`rand(prior, n)` when `prior = intrinsic_prior(FullBNS())`:

- `mass::Matrix{Float64}` of size `(2, n)`; row 1 is `mass_1_source`, row 2 is `mass_2_source`.
- `redshift`, `χ₁`, `χ₂`, `Λ₁`, `Λ₂` are `Vector{Float64}` of length `n`.

HDF5 proposal columns remain ASCII (`chi_1`, `lambda_1`, …). Matches `keys(prior.dists)` for the full-BNS intrinsic prior.
"""
const FullBNSSamplesSoA = @NamedTuple{
    mass::Matrix{Float64},
    redshift::Vector{Float64},
    χ₁::Vector{Float64},
    χ₂::Vector{Float64},
    Λ₁::Vector{Float64},
    Λ₂::Vector{Float64}
}

"""
    stack_source_masses(mass_1_source, mass_2_source) -> Matrix{Float64}

Pack two same-length mass vectors into the `2 × n` matrix expected by
[`FullBNSSamplesSoA`](@ref)`.mass` (row 1 = `mass_1_source`, row 2 = `mass_2_source`).
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

"""Canonical ordering of full-BNS intrinsic columns on disk and inside proposal vectors."""
const FULL_BNS_INTRINSIC_ORDER = [
    "mass_1_source", "mass_2_source", "redshift", "chi_1", "chi_2", "lambda_1", "lambda_2"]

"""
    resolve_intrinsic_strategy(intrinsic_site_order::Vector{String}) -> FullBNS

Map a proposal `intrinsic_site_order` to a concrete [`IntrinsicPriorStrategy`](@ref).
Currently only the canonical [`FULL_BNS_INTRINSIC_ORDER`](@ref) is supported and the
function returns a [`FullBNS`](@ref) singleton; anything else throws `ArgumentError`.
"""
function resolve_intrinsic_strategy(intrinsic_site_order::Vector{String})::FullBNS
    if intrinsic_site_order == FULL_BNS_INTRINSIC_ORDER
        return FullBNS()
    end
    throw(
        ArgumentError(
        "unsupported intrinsic_site_order $(repr(intrinsic_site_order)); " *
        "only the full BNS layout is supported: $(repr(FULL_BNS_INTRINSIC_ORDER)). " *
        "Redshift-only caches are no longer supported.",
    ),
    )
end
