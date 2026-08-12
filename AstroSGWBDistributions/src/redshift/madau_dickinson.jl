export madau_dickinson_source_frame_distribution,
       MadauDickinsonSourceFrame, source_frame_distribution, redshift_prior

"""
    madau_dickinson_source_frame_distribution(z; γ, κ, zpeak) -> Real

Source-frame merger-rate density at redshift `z` under the Madau–Dickinson model.
The denominator exponent is `γ + κ` (so `κ` is the increment beyond `γ`).
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
    MadauDickinsonSourceFrame(; γ, κ, zpeak)

Parameterized Madau–Dickinson (2014) source-frame merger-rate model.
"""
struct MadauDickinsonSourceFrame{Tγ <: Real, Tκ <: Real, Tzpeak <: Real}
    γ::Tγ
    κ::Tκ
    zpeak::Tzpeak
end

function MadauDickinsonSourceFrame(; γ::Real, κ::Real, zpeak::Real)
    γ′, κ′, zpeak′ = promote(γ, κ, zpeak)
    return MadauDickinsonSourceFrame(γ′, κ′, zpeak′)
end

"""
    source_frame_distribution(model::MadauDickinsonSourceFrame, z) -> Real

Source-frame merger-rate density at redshift `z` under the Madau–Dickinson model.
The denominator exponent is `γ + κ`.
"""
function source_frame_distribution(model::MadauDickinsonSourceFrame, z::Real)
    return madau_dickinson_source_frame_distribution(
        z; γ = model.γ, κ = model.κ, zpeak = model.zpeak)
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
    return RedshiftInterpolatedDistribution(build_redshift_prior(sfn, cosmo, z_grid))
end
