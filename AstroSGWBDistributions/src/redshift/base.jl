using Distributions
using Random

export RedshiftPrior, redshift_integral, redshift_log_prob, merger_rate_per_sec,
       detector_frame_merger_rate_density, expected_number_of_events,
       build_redshift_prior,
       RedshiftInterpolatedDistribution,
       DEFAULT_Z_GRID

"""
    DEFAULT_Z_GRID

Default redshift integration grid: 256 uniformly-spaced points on [0, 20].
Shared across [`redshift_prior`](@ref) calls that do not pass an explicit grid.

The grid must start at `0` and be strictly increasing. [`distance_and_volume_grid`](@ref)
enforces these requirements because its cumulative comoving-distance integral assumes
`d_c(0) = 0`.
"""
const DEFAULT_Z_GRID = collect(LinRange(0.0, 20.0, 256))

"""
    RedshiftPrior(x, y, interpolant)

Domain wrapper for the detector-frame merger-rate density grid and its linear
interpolant. Cumulative values are computed on demand for inverse-CDF sampling.
"""
struct RedshiftPrior{I, TX <: AbstractVector, TY <: AbstractVector}
    x::TX
    y::TY
    itp::I
end

"""
    redshift_integral(prior::RedshiftPrior) -> Real

Detector-frame redshift-integrated merger-rate density on the grid.
"""
redshift_integral(prior::RedshiftPrior) = trapz(prior.x, prior.y)

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

Public API for the sampling and diagnostic paths. It no longer sits on any inference
path: `merger_rate_per_sec` used to route through it, and `T` cancelled -- see S7.
"""
function expected_number_of_events(
        local_merger_rate_gpc3_yr::Real,
        redshift_integral_mpc3::Real,
        observation_time::Real
)
    return 1e-9 * local_merger_rate_gpc3_yr * redshift_integral_mpc3 * observation_time
end

"""
    merger_rate_per_sec(redshift_integral_mpc3, local_merger_rate_gpc3_yr)
    merger_rate_per_sec(prior::RedshiftPrior, local_merger_rate_gpc3_yr)

Detector-frame merger rate in events/sec:
`1e-9 · R₀ · ∫dN/dz / JULIAN_YEAR_SEC`.

There is deliberately no `observation_time` argument. The rate is a rate; the old
three-argument form computed
`expected_number_of_events(R₀, ∫, T) / year_to_second(T) = 1e-9·R₀·∫·T / (T·JULIAN_YEAR_SEC)`,
where `T` cancels algebraically -- it was multiplied and divided by itself. Detector
state, `observation_time` included, does not belong on the importance-weighting path.
[`expected_number_of_events`](@ref) keeps its `T`, which it genuinely uses.

The scalar form is what the importance-weighting hot path calls: it needs only the
redshift integral, so it does not have to build a [`RedshiftPrior`](@ref). The
`RedshiftPrior` form computes its integral on demand and stays the entry point for
the sampling path.
"""
function merger_rate_per_sec(
        redshift_integral_mpc3::Real,
        local_merger_rate_gpc3_yr::Real
)
    return 1.0e-9 * local_merger_rate_gpc3_yr * redshift_integral_mpc3 / JULIAN_YEAR_SEC
end

function merger_rate_per_sec(prior::RedshiftPrior, local_merger_rate_gpc3_yr::Real)
    return merger_rate_per_sec(redshift_integral(prior), local_merger_rate_gpc3_yr)
end

"""
    build_redshift_prior(source_frame_fn, cosmology, z_grid) -> RedshiftPrior

Tabulate the detector-frame redshift density on the caller's grid. Distances and
comoving volume come from [`distance_and_volume_grid`](@ref); the returned
[`RedshiftPrior`](@ref) stores the grid, density values, and linear interpolant;
cumulative values are computed on demand for inverse-CDF sampling.
"""
function build_redshift_prior(
        source_frame_fn,
        cosmo::AbstractCosmology,
        z_grid::AbstractVector{<:Real}
)
    z_grid_f = z_grid isa AbstractVector{Float64} ? z_grid : collect(Float64, z_grid)
    grid = distance_and_volume_grid(cosmo, z_grid_f)
    pdf_vals = map(eachindex(z_grid_f)) do i
        @inbounds detector_frame_merger_rate_density(
            z_grid_f[i],
            grid.differential_comoving_volume[i],
            source_frame_fn(z_grid_f[i])
        )
    end
    return RedshiftPrior(z_grid_f, pdf_vals, LinearInterpolation(pdf_vals, z_grid_f))
end

@inline function _normalized_log_density(pdf_at_value, norm, tiny)
    return log(max(pdf_at_value / max(norm, tiny), tiny))
end

function redshift_log_prob(prior::RedshiftPrior, value::Real)
    norm = redshift_integral(prior)
    T = promote_type(eltype(prior.y), typeof(norm))
    tiny = floatmin(T)
    pdf_at_value = prior.itp(value)
    return _normalized_log_density(pdf_at_value, norm, tiny)
end

"""
    redshift_logpdf_eltype(prior::RedshiftPrior) -> Type

Element type of values returned by the redshift log-density associated with
`prior`. Useful for preallocating output vectors that promote with the
redshift contribution (for example `ForwardDiff.Dual` when `prior` was built
under AD).
"""
function redshift_logpdf_eltype(prior::RedshiftPrior)
    return promote_type(eltype(prior.y), typeof(redshift_integral(prior)))
end

struct RedshiftInterpolatedDistribution{P <: RedshiftPrior} <:
       ContinuousUnivariateDistribution
    prior::P
end

Base.minimum(d::RedshiftInterpolatedDistribution) = first(d.prior.x)
Base.maximum(d::RedshiftInterpolatedDistribution) = last(d.prior.x)
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
    cumulative = cumtrapz(d.prior.x, d.prior.y)
    x = d.prior.x
    n = length(cumulative)
    idx = searchsortedlast(cumulative, target)
    idx <= 0 && return x[1]
    idx >= n && return x[end]
    c0, c1 = cumulative[idx], cumulative[idx + 1]
    x0, x1 = x[idx], x[idx + 1]
    c1 > c0 || return x0
    return x0 + (target - c0) * (x1 - x0) / (c1 - c0)
end
