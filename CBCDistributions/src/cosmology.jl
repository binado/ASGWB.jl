using QuadGK

const SPEED_OF_LIGHT_KM_S = 299792.458

"""1 Mpc in meters (IAU/CODATA convention)."""
const METERS_PER_MPC = 3.085677581e22

"""
    hubble_constant_si(H0_km_s_mpc::Real) -> Float64

Hubble constant in s⁻¹ from ``H_0`` in **km/s/Mpc** (same units as [`Cosmology`](@ref).`H0`).
"""
function hubble_constant_si(H0_km_s_mpc::Real)
    return Float64(H0_km_s_mpc) * 1000.0 / METERS_PER_MPC
end

struct Cosmology{TH0 <: Real, TΩm <: Real}
    H0::TH0
    Ωm::TΩm
end

Cosmology(h::NamedTuple) = Cosmology(h.H0, h.Ωm)

struct CosmologyCache{C <: Cosmology, I <: CumulativeIntegral1D, TD <: Real}
    cosmology::C
    inv_E_integral::I
    d_h::TD
end

function CosmologyCache(cosmology::Cosmology, z_grid::AbstractVector{<:Real})
    inv_E_integral = CumulativeIntegral1D(z_grid, z -> inv(E(z, cosmology.Ωm)))
    d_h = SPEED_OF_LIGHT_KM_S / cosmology.H0
    return CosmologyCache(cosmology, inv_E_integral, d_h)
end

function CosmologyCache(h::NamedTuple, z_grid::AbstractVector{<:Real})
    CosmologyCache(Cosmology(h), z_grid)
end

function E(z::Real, Ωm::Real)
    return sqrt(Ωm * (1 + z)^3 + (1 - Ωm))
end

function comoving_distance(z::Real, H0::Real, Ωm::Real)
    z == zero(z) && return zero(float(promote_type(typeof(z), typeof(H0), typeof(Ωm))))
    integral, _ = quadgk(x -> inv(E(x, Ωm)), zero(z), z)
    return (SPEED_OF_LIGHT_KM_S / H0) * integral
end

function luminosity_distance(z::Real, H0::Real, Ωm::Real)
    (1 + z) * comoving_distance(z, H0, Ωm)
end

function differential_comoving_volume(z::Real, H0::Real, Ωm::Real)
    d_h = SPEED_OF_LIGHT_KM_S / H0
    d_c = comoving_distance(z, H0, Ωm)
    return d_h * d_c^2 / E(z, Ωm)
end

function comoving_distance(z::Real, cache::CosmologyCache)
    cache.d_h * cdf(cache.inv_E_integral, z)
end

function luminosity_distance(z::Real, cache::CosmologyCache)
    (1 + z) * comoving_distance(z, cache)
end

function differential_comoving_volume(z::Real, cache::CosmologyCache)
    d_c = comoving_distance(z, cache)
    return cache.d_h * d_c^2 / E(z, cache.cosmology.Ωm)
end

"""
    comoving_distance(z, H0, Ωm, dist::CumulativeIntegral1D) -> Real

Comoving distance using a precomputed [`CumulativeIntegral1D`](@ref) of
`w -> 1/E(w, Ωm)`. Uses [`cdf`](@ref) which returns the exact integral under
the linear interpolant (analytic trapezoidal rule).
"""
function comoving_distance(z::Real, H0::Real, Ωm::Real, dist::CumulativeIntegral1D)
    (SPEED_OF_LIGHT_KM_S / H0) * cdf(dist, z)
end

function luminosity_distance(z::Real, H0::Real, Ωm::Real, dist::CumulativeIntegral1D)
    (1 + z) * comoving_distance(z, H0, Ωm, dist)
end

function differential_comoving_volume(
        z::Real,
        H0::Real,
        Ωm::Real,
        dist::CumulativeIntegral1D
)
    d_h = SPEED_OF_LIGHT_KM_S / H0
    d_c = comoving_distance(z, H0, Ωm, dist)
    return d_h * d_c^2 / E(z, Ωm)
end

function gravitational_wave_distance(
        z::Real,
        luminosity_distance::Real,
        Ξ₀::Real,
        Ξₙ::Real
)
    return (Ξ₀ + (1 - Ξ₀) / (1 + z)^Ξₙ) * luminosity_distance
end
