using Distributions
using Random

export RedshiftPrior, redshift_integral, redshift_log_prob, merger_rate_per_sec,
       detector_frame_merger_rate_density, expected_number_of_events,
       madau_dickinson_source_frame_distribution,
       build_redshift_prior,
       RedshiftInterpolatedDistribution,
       MadauDickinsonSourceFrame, source_frame_distribution, redshift_prior,
       DEFAULT_Z_GRID

"""
    DEFAULT_Z_GRID

Default redshift integration grid: 256 uniformly-spaced points on [0, 20].
Shared across [`redshift_prior`](@ref) calls that do not pass an explicit grid.

**The grid must start at `0`.** Comoving distance is accumulated by trapezoidal
integration along the grid assuming `d_c(z_grid[1]) = 0`, so a non-zero lower bound
silently omits `∫₀^{z_min} dz/E` from *every* distance. The previous `1e-3` lower bound
cost ≈ 4.5 Mpc — −1.0% in `d_L` at z = 0.1 and −20% at z = 0.005 — which is also a factor
of `1/z` in the relative interpolation error at low redshift. It matches
`astrogwb.cosmology.distance_and_volume_grid`, which carries the same requirement.
"""
const DEFAULT_Z_GRID = collect(LinRange(0.0, 20.0, 256))

"""
    RedshiftPrior(dN_dz)

Domain wrapper for the detector-frame merger-rate density cumulative integral.
Its
  [`normalizer`](@ref) is the redshift integral ``∫ p(z)\\,dz`` driving
  [`merger_rate_per_sec`](@ref) and its cumulative table supports inverse-CDF
  sampling in [`RedshiftInterpolatedDistribution`](@ref).
"""
struct RedshiftPrior{P <: CumulativeIntegral1D}
    dN_dz::P
end

"""
    redshift_integral(prior::RedshiftPrior) -> Real

Convenience wrapper for `normalizer(prior.dN_dz)` — the detector-frame
redshift-integrated merger-rate density on the grid.
"""
redshift_integral(prior::RedshiftPrior) = normalizer(prior.dN_dz)

function detector_frame_merger_rate_density(
        z::Real,
        differential_comoving_volume::Real,
        source_frame_distribution::Real
)
    return 4π * differential_comoving_volume * source_frame_distribution / (1 + z)
end

"""
    expected_number_of_events(local_merger_rate_gpc3_yr, redshift_integral_mpc3, observation_time) -> Real

Expected number of detected events over the observation.

`observation_time` is the observation duration in years (Julian year).
"""
function expected_number_of_events(
        local_merger_rate_gpc3_yr::Real,
        redshift_integral_mpc3::Real,
        observation_time::Real
)
    return 1e-9 * local_merger_rate_gpc3_yr * redshift_integral_mpc3 * observation_time
end

"""
    merger_rate_per_sec(redshift_integral_mpc3, local_merger_rate_gpc3_yr, observation_time)
    merger_rate_per_sec(prior::RedshiftPrior, local_merger_rate_gpc3_yr, observation_time)

Detector-frame merger rate in events/sec:
`expected_number_of_events(local_rate, redshift_integral, observation_time) /
year_to_second(observation_time)`.

The scalar form is what the importance-weighting hot path calls: it needs only the
redshift integral, so it does not have to build a [`RedshiftPrior`](@ref) just to read
`normalizer` back out of it. The `RedshiftPrior` form forwards to it and stays the entry
point for the sampling path.

`observation_time` is the observation duration in years (Julian year).
"""
function merger_rate_per_sec(
        redshift_integral_mpc3::Real,
        local_merger_rate_gpc3_yr::Real,
        observation_time::Real
)
    n_events = expected_number_of_events(
        local_merger_rate_gpc3_yr,
        redshift_integral_mpc3,
        observation_time
    )
    return n_events / year_to_second(observation_time)
end

function merger_rate_per_sec(
        prior::RedshiftPrior,
        local_merger_rate_gpc3_yr::Real,
        observation_time::Real
)
    return merger_rate_per_sec(
        redshift_integral(prior), local_merger_rate_gpc3_yr, observation_time)
end

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
    MadauDickinsonSourceFrame

Dispatch tag for the Madau–Dickinson (2014) star-formation-rate source-frame
merger-rate model.  Pass to [`source_frame_distribution`](@ref) or
[`redshift_prior`](@ref).
"""
struct MadauDickinsonSourceFrame end

"""
    source_frame_distribution(::MadauDickinsonSourceFrame, z, Λ) -> Real

Source-frame merger-rate density at redshift `z` under the Madau–Dickinson model.
Reads `γ`, `κ`, `zpeak` from `Λ`; the denominator exponent is `γ + κ`.
"""
function source_frame_distribution(::MadauDickinsonSourceFrame, z::Real, Λ::NamedTuple)
    return madau_dickinson_source_frame_distribution(z; γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak)
end

"""
    redshift_prior(sf_model, cache::CosmologyCache, Λ) -> RedshiftInterpolatedDistribution

Build the detector-frame redshift prior from a prebuilt [`CosmologyCache`](@ref),
reusing its cumulative ∫1/E integral (and grid) rather than recomputing them. This
is the form the hot path calls so the same cache is shared with importance
weighting instead of being rebuilt per evaluation.
"""
function redshift_prior(sf_model, cache::CosmologyCache, Λ::NamedTuple)
    sfn = z -> source_frame_distribution(sf_model, z, Λ)
    return RedshiftInterpolatedDistribution(build_redshift_prior(sfn, cache))
end

"""
    redshift_prior(sf_model, cosmo, Λ; z_grid) -> RedshiftInterpolatedDistribution

Convenience form that builds a [`CosmologyCache`](@ref) on `z_grid` (default
[`DEFAULT_Z_GRID`](@ref)) and delegates to the cache method. Use when no cache is
already on hand.
"""
function redshift_prior(
        sf_model,
        cosmo::AbstractCosmology,
        Λ::NamedTuple;
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
)
    return redshift_prior(sf_model, CosmologyCache(cosmo, z_grid), Λ)
end

function build_redshift_prior(source_frame_fn, cache::CosmologyCache)
    z_grid_f = cache.inv_E_integral.x
    pdf_vals = map(eachindex(z_grid_f)) do i
        @inbounds z = z_grid_f[i]
        @inbounds d_c = cache.d_h * cache.inv_E_integral.cumulative[i]
        dvc_dz = cache.d_h * d_c^2 / E(z, cache.cosmology)
        detector_frame_merger_rate_density(z, dvc_dz, source_frame_fn(z))
    end
    return RedshiftPrior(CumulativeIntegral1D(z_grid_f, pdf_vals))
end

@inline function _normalized_log_density(pdf_at_value, norm, tiny)
    return log(max(pdf_at_value / max(norm, tiny), tiny))
end

function redshift_log_prob(prior::RedshiftPrior, value::Real)
    norm = redshift_integral(prior)
    T = promote_type(eltype(prior.dN_dz.y), typeof(norm))
    tiny = floatmin(T)
    pdf_at_value = interpolate(prior.dN_dz, value)
    return _normalized_log_density(pdf_at_value, norm, tiny)
end

@inline function _redshift_logpdf(prior::RedshiftPrior, z::Real)
    x_lo = first(prior.dN_dz.x)
    x_hi = last(prior.dN_dz.x)
    (z < x_lo || z > x_hi) && return -Inf
    return redshift_log_prob(prior, z)
end

"""
    redshift_logpdf_eltype(prior::RedshiftPrior) -> Type

Element type of values returned by the redshift log-density associated with
`prior`. Useful for preallocating output vectors that promote with the
redshift contribution (for example `ForwardDiff.Dual` when `prior` was built
under AD).
"""
function redshift_logpdf_eltype(prior::RedshiftPrior)
    return promote_type(eltype(prior.dN_dz.y), typeof(redshift_integral(prior)))
end

struct RedshiftInterpolatedDistribution{P <: RedshiftPrior} <:
       ContinuousUnivariateDistribution
    prior::P
end

Base.minimum(d::RedshiftInterpolatedDistribution) = first(d.prior.dN_dz.x)
Base.maximum(d::RedshiftInterpolatedDistribution) = last(d.prior.dN_dz.x)
Base.eltype(d::RedshiftInterpolatedDistribution) = redshift_logpdf_eltype(d.prior)

function Distributions.insupport(d::RedshiftInterpolatedDistribution, value::Real)
    return minimum(d) <= value <= maximum(d)
end

function Distributions.logpdf(d::RedshiftInterpolatedDistribution, value::Real)
    insupport(d, value) || return -Inf
    return redshift_log_prob(d.prior, value)
end

function Random.rand(rng::AbstractRNG, d::RedshiftInterpolatedDistribution)
    target = rand(rng) * redshift_integral(d.prior)
    cumulative = d.prior.dN_dz.cumulative
    x = d.prior.dN_dz.x
    n = length(cumulative)
    idx = searchsortedlast(cumulative, target)
    idx <= 0 && return x[1]
    idx >= n && return x[end]
    c0, c1 = cumulative[idx], cumulative[idx + 1]
    x0, x1 = x[idx], x[idx + 1]
    c1 > c0 || return x0
    return x0 + (target - c0) * (x1 - x0) / (c1 - c0)
end
