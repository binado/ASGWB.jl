export madau_dickinson_source_frame_distribution,
       MadauDickinsonSourceFrame, source_frame_distribution, redshift_prior

"""
    madau_dickinson_source_frame_distribution(z; γ, κ, zpeak) -> Real

Source-frame merger-rate **shape** at redshift `z` under the Madau–Dickinson model
(normalized so the density is 1 at `z = 0`). The denominator exponent is `γ + κ`
(so `κ` is the increment beyond `γ`).

Absolute amplitude (`R₀` and unit conversions) lives on
[`MadauDickinsonSourceFrame`](@ref) / [`source_frame_distribution`](@ref).
"""
function madau_dickinson_source_frame_distribution(
        z::Real;
        γ::Real,
        κ::Real,
        zpeak::Real
)
    one_plus_z = 1 + z
    denom_exp = γ + κ
    return ((one_plus_z^γ) / (1 + (one_plus_z / (1 + zpeak))^denom_exp)) *
           (1 + (1 + zpeak)^(-denom_exp))
end

# ---------------------------------------------------------------------------
# Redshift prior seam: dispatch on source-frame model type
# ---------------------------------------------------------------------------

"""
    MadauDickinsonSourceFrame(; γ, κ, zpeak, R₀)

Parameterized Madau–Dickinson (2014) source-frame merger-rate model, including the local
merger rate `R₀` (Gpc⁻³ yr⁻¹).
"""
struct MadauDickinsonSourceFrame{
    Tγ <: Real, Tκ <: Real, Tzpeak <: Real, TR₀ <: Real
}
    γ::Tγ
    κ::Tκ
    zpeak::Tzpeak
    R₀::TR₀
end

function MadauDickinsonSourceFrame(; γ::Real, κ::Real, zpeak::Real, R₀::Real)
    γ′, κ′, zpeak′, R₀′ = promote(γ, κ, zpeak, R₀)
    return MadauDickinsonSourceFrame(γ′, κ′, zpeak′, R₀′)
end

"""
    source_frame_distribution(model::MadauDickinsonSourceFrame, z) -> Real

Source-frame merger-rate density at redshift `z`, including the local rate and the
Gpc³→Mpc³ / yr→sec conversions so that the detector-frame
[`normalizer`](@ref) is events/sec:

`(1e-9 · R₀ / JULIAN_YEAR_SEC) · ψ_shape(z)`.
"""
function source_frame_distribution(model::MadauDickinsonSourceFrame, z::Real)
    shape = madau_dickinson_source_frame_distribution(
        z; γ = model.γ, κ = model.κ, zpeak = model.zpeak)
    return (1.0e-9 * model.R₀ / JULIAN_YEAR_SEC) * shape
end

"""
    redshift_prior(model, cosmology; z_grid) -> RedshiftInterpolatedDistribution

Build the detector-frame redshift distribution on `z_grid` (default
[`DEFAULT_Z_GRID`](@ref)). The cosmology package tabulates distance and volume; this
module owns the redshift density, normalization, and inverse-CDF sampling state.
"""
function redshift_prior(
        model::MadauDickinsonSourceFrame,
        cosmo::AbstractCosmology,
        ;
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
)
    sfn = z -> source_frame_distribution(model, z)
    return RedshiftInterpolatedDistribution(sfn, cosmo, z_grid)
end
