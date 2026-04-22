"""
    spectral_density(fluxes, merger_rate_per_sec; weights=nothing) -> Vector

Collapse per-sample flux contributions into a spectral density vector.

`fluxes` is a `(n_freq, n_samples)` matrix (column-major friendly). When `weights`
is `nothing`, samples are averaged uniformly: `mean_flux = sum(fluxes; dims=2) / n_samples`.
When `weights` is supplied, the contraction is `fluxes * weights / n_samples`
(no normalization of `weights`). The `0.4 = 2/5` prefactor captures the
average over the inclination angle.
"""
function spectral_density(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real;
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing
)
    n_samples = size(fluxes, 2)
    mean_flux = if weights === nothing
        vec(sum(fluxes; dims = 2)) ./ n_samples
    else
        length(weights) == n_samples || throw(
            ArgumentError(
            "weights length ($(length(weights))) must match fluxes sample count ($(n_samples))",
        ),
        )
        (fluxes * weights) ./ n_samples
    end
    return 0.4 .* merger_rate_per_sec .* mean_flux
end

"""1 Mpc in meters (IAU/CODATA convention)."""
const METERS_PER_MPC = 3.085677581e22

"""
    hubble_constant_si(H0_km_s_mpc::Real) -> Float64

Hubble constant in s⁻¹ from ``H_0`` in **km/s/Mpc** (same units as [`HyperParameters`](@ref).`H0`).
"""
function hubble_constant_si(H0_km_s_mpc::Real)
    return Float64(H0_km_s_mpc) * 1000.0 / METERS_PER_MPC
end

"""
    omegagw(spectral_density, frequency, H0::Real)
    omegagw(spectral_density, frequency, parameters::HyperParameters)

Dimensionless gravitational-wave energy density per logarithmic frequency,

``\\Omega_{\\mathrm{GW}}(f) = \\frac{4\\pi^2}{3 H_0^2} f^3 S_h(f)``,

where ``S_h(f)`` is the strain spectral density (same units as [`spectral_density`](@ref) on fluxes)
and ``H_0`` is the Hubble constant in **s⁻¹**.

The `H0::Real` method takes ``H_0`` in **km/s/Mpc** (matching the rest of this package) and converts
it internally to s⁻¹. The [`HyperParameters`](@ref) method uses `parameters.H0`.

`frequency` and `spectral_density` may be scalars or arrays; they broadcast together (e.g. same-length
vectors for one spectrum per frequency bin).
"""
function omegagw(spectral_density, frequency, H0::Real)
    h0_si = hubble_constant_si(H0)
    pre = 4 * pi^2 / (3 * h0_si^2)
    return @. pre * frequency^3 * spectral_density
end

function omegagw(spectral_density, frequency, parameters::HyperParameters)
    omegagw(spectral_density, frequency, parameters.H0)
end
