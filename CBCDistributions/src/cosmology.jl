using QuadGK

const SPEED_OF_LIGHT_KM_S = 299792.458

"""1 Mpc in meters (IAU/CODATA convention)."""
const METERS_PER_MPC = 3.085677581e22

"""
    hubble_constant_si(H0_km_s_mpc::Real) -> Float64

Hubble constant in s⁻¹ from ``H_0`` in **km/s/Mpc** (same units as [`LambdaCDM`](@ref).`H0`).
"""
function hubble_constant_si(H0_km_s_mpc::Real)
    return Float64(H0_km_s_mpc) * 1000.0 / METERS_PER_MPC
end

"""Abstract supertype for flat FLRW cosmology models."""
abstract type AbstractCosmology end

"""Flat ΛCDM cosmology (w=-1, radiation-free)."""
struct LambdaCDM{TH0 <: Real, TΩm <: Real} <: AbstractCosmology
    H0::TH0
    Ωm::TΩm
end

"""Flat wCDM cosmology with constant dark-energy equation of state w0."""
struct W0CDM{TH0 <: Real, TΩm <: Real, Tw0 <: Real} <: AbstractCosmology
    H0::TH0
    Ωm::TΩm
    w0::Tw0
end

"""Flat w0waCDM (CPL) cosmology: w(z) = w0 + wa·z/(1+z)."""
struct W0WaCDM{TH0 <: Real, TΩm <: Real, Tw0 <: Real, Twa <: Real} <: AbstractCosmology
    H0::TH0
    Ωm::TΩm
    w0::Tw0
    wa::Twa
end

H0(c::AbstractCosmology) = c.H0
Ωm(c::AbstractCosmology) = c.Ωm

Base.broadcastable(c::AbstractCosmology) = Ref(c)

"""Supported flat-FLRW cosmology subtypes (registration order)."""
const SUPPORTED_COSMOLOGIES = (LambdaCDM, W0CDM, W0WaCDM)

"""
    cosmology_parameters(::Type{C}) -> Tuple{Vararg{Symbol}}

Hyperparameter symbols for cosmology subtype `C`, in struct field order.
"""
cosmology_parameters(::Type{C}) where {C <: AbstractCosmology} = fieldnames(C)

"""
    cosmology(::Type{C}, h::NamedTuple) -> C

Build cosmology subtype `C` from hyperparameter state `h` (fields must match [`cosmology_parameters`](@ref)(`C`)).
"""
function cosmology(::Type{C}, h::NamedTuple) where {C <: AbstractCosmology}
    fn = fieldnames(C)
    return C(ntuple(i -> h[fn[i]], Val(length(fn)))...)
end

"""
    cosmology(h::NamedTuple) -> AbstractCosmology

Infer cosmology subtype from keys in `h` (`:wa` → [`W0WaCDM`](@ref), `:w0` → [`W0CDM`](@ref), else [`LambdaCDM`](@ref)).
"""
function cosmology(h::NamedTuple)
    :wa in keys(h) && return cosmology(W0WaCDM, h)
    :w0 in keys(h) && return cosmology(W0CDM, h)
    return cosmology(LambdaCDM, h)
end

function (::Type{C})(h::NamedTuple) where {C <: AbstractCosmology}
    return cosmology(C, h)
end

cosmology_config_name(::Type{LambdaCDM}) = "LambdaCDM"
cosmology_config_name(::Type{W0CDM}) = "W0CDM"
cosmology_config_name(::Type{W0WaCDM}) = "W0WaCDM"

const _COSMOLOGY_BY_CONFIG_NAME = Dict(
    cosmology_config_name(C) => C for C in SUPPORTED_COSMOLOGIES
)

"""
    cosmology_type(name::AbstractString) -> Type{<:AbstractCosmology}

Resolve a config/TOML cosmology name to a concrete subtype.
"""
function cosmology_type(name::AbstractString)
    C = get(_COSMOLOGY_BY_CONFIG_NAME, String(name), nothing)
    C === nothing && throw(
        ArgumentError(
        "unknown cosmology \"$(name)\"; valid choices: $(sort(collect(keys(_COSMOLOGY_BY_CONFIG_NAME))))",
    ),
    )
    return C
end

"""
    dark_energy_eos(c::AbstractCosmology, z) -> Real

Dark energy equation of state w(z).
"""
dark_energy_eos(::LambdaCDM, z) = -one(z)
dark_energy_eos(c::W0CDM, z) = c.w0
dark_energy_eos(c::W0WaCDM, z) = c.w0 + c.wa * z / (1 + z)

"""
    de_density_ratio(c::AbstractCosmology, z) -> Real

Ratio ρ_DE(z)/ρ_DE(0): closed-form integral of `dark_energy_eos` through the Friedmann equation.
"""
de_density_ratio(::LambdaCDM, z) = one(z)
de_density_ratio(c::W0CDM, z) = (1 + z)^(3 * (1 + c.w0))
function de_density_ratio(c::W0WaCDM, z)
    (1 + z)^(3 * (1 + c.w0 + c.wa)) * exp(-3 * c.wa * z / (1 + z))
end

"""
    E(z, c::AbstractCosmology) -> Real

Hubble parameter ratio E(z) = H(z)/H₀ for flat FLRW cosmology.
"""
function E(z::Real, c::AbstractCosmology)
    return sqrt(Ωm(c) * (1 + z)^3 + (1 - Ωm(c)) * de_density_ratio(c, z))
end

struct CosmologyCache{C <: AbstractCosmology, I <: CumulativeIntegral1D, TD <: Real}
    cosmology::C
    inv_E_integral::I
    d_h::TD
end

function CosmologyCache(cosmology::AbstractCosmology, z_grid::AbstractVector{<:Real})
    inv_E_integral = CumulativeIntegral1D(z_grid, z -> inv(E(z, cosmology)))
    d_h = SPEED_OF_LIGHT_KM_S / H0(cosmology)
    return CosmologyCache(cosmology, inv_E_integral, d_h)
end

function comoving_distance(z::Real, c::AbstractCosmology)
    Ez = E(z, c)
    pref = SPEED_OF_LIGHT_KM_S / (H0(c) * Ez)
    z == zero(z) && return zero(pref)
    integral, _ = quadgk(x -> inv(E(x, c)), zero(z), z)
    return pref * integral * Ez
end

function luminosity_distance(z::Real, c::AbstractCosmology)
    (1 + z) * comoving_distance(z, c)
end

function differential_comoving_volume(z::Real, c::AbstractCosmology)
    d_h = SPEED_OF_LIGHT_KM_S / H0(c)
    d_c = comoving_distance(z, c)
    return d_h * d_c^2 / E(z, c)
end

function comoving_distance(z::Real, cache::CosmologyCache)
    cache.d_h * cdf(cache.inv_E_integral, z)
end

function luminosity_distance(z::Real, cache::CosmologyCache)
    (1 + z) * comoving_distance(z, cache)
end

function differential_comoving_volume(z::Real, cache::CosmologyCache)
    d_c = comoving_distance(z, cache)
    return cache.d_h * d_c^2 / E(z, cache.cosmology)
end

"""
    comoving_distance(z, c::AbstractCosmology, dist::CumulativeIntegral1D) -> Real

Comoving distance using a precomputed [`CumulativeIntegral1D`](@ref) of
`w -> 1/E(w, c)`. Uses [`cdf`](@ref) which returns the exact integral under
the linear interpolant (analytic trapezoidal rule).
"""
function comoving_distance(z::Real, c::AbstractCosmology, dist::CumulativeIntegral1D)
    (SPEED_OF_LIGHT_KM_S / H0(c)) * cdf(dist, z)
end

function luminosity_distance(z::Real, c::AbstractCosmology, dist::CumulativeIntegral1D)
    (1 + z) * comoving_distance(z, c, dist)
end

function differential_comoving_volume(z::Real, c::AbstractCosmology, dist::CumulativeIntegral1D)
    d_h = SPEED_OF_LIGHT_KM_S / H0(c)
    d_c = comoving_distance(z, c, dist)
    return d_h * d_c^2 / E(z, c)
end

function gravitational_wave_distance(
        z::Real,
        luminosity_distance::Real,
        Ξ₀::Real,
        Ξₙ::Real
)
    return (Ξ₀ + (1 - Ξ₀) / (1 + z)^Ξₙ) * luminosity_distance
end
