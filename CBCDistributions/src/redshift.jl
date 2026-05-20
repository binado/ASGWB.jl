using Distributions
using Random

export RedshiftPrior, redshift_integral, redshift_log_prob, merger_rate_per_sec,
       detector_frame_merger_rate_density, expected_number_of_events,
       madau_dickinson_source_frame_distribution, power_law_source_frame_distribution,
       redshift_grid, SampleInterpolant, _interpolate_at_sample, _cdf_at_sample,
       luminosity_distance_at_sample,
       build_redshift_prior, cosmology_and_redshift_prior,
       RedshiftInterpolatedDistribution, _normalized_log_density,
       redshift_log_prob_samples, redshift_log_prob_samples!, redshift_logpdf_eltype

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

function expected_number_of_events(
        local_merger_rate_gpc3_yr::Real,
        redshift_integral_mpc3::Real,
        observation_time_yr::Real
)
    return 1e-9 * local_merger_rate_gpc3_yr * redshift_integral_mpc3 * observation_time_yr
end

"""
    merger_rate_per_sec(
        prior, local_merger_rate_gpc3_yr, observation_time_yr, observation_time_sec,
    ) -> Float64

Detector-frame merger rate in events/sec:
`expected_number_of_events(local_rate, redshift_integral(prior), observation_time_yr) /
observation_time_sec`. `observation_time_yr` sets the events count, `observation_time_sec`
converts to per-second units; they are taken independently rather than assuming a fixed
seconds-per-year so the cache's stored pair of times round-trips exactly.
"""
function merger_rate_per_sec(
        prior::RedshiftPrior,
        local_merger_rate_gpc3_yr::Real,
        observation_time_yr::Real,
        observation_time_sec::Real
)
    n_events = expected_number_of_events(
        local_merger_rate_gpc3_yr,
        redshift_integral(prior),
        observation_time_yr
    )
    return n_events / observation_time_sec
end

function madau_dickinson_source_frame_distribution(
        z::Real;
        γ::Real,
        κ::Real,
        zpeak::Real
)
    one_plus_z = 1 + z
    return ((one_plus_z^γ) / (1 + (one_plus_z / (1 + zpeak))^κ)) *
           (1 + (1 + zpeak)^(-κ))
end

power_law_source_frame_distribution(z::Real; Λ::Real) = (1 + z)^Λ

"""
    redshift_grid(spec::RedshiftPriorSpec) -> Vector{Float64}

Uniform redshift grid implied by `spec` (materialized as `Vector{Float64}`).
This is safe to precompute once and reuse across likelihood evaluations.
"""
function redshift_grid(spec::RedshiftPriorSpec)
    return collect(LinRange(spec.z_min, spec.z_max, spec.num_interp))
end

"""
    SampleInterpolant

Per-sample interpolation metadata for proposal redshifts on the fixed redshift
grid. `bin_idx[i]` is the lower grid cell index for sample `i`; `t[i]` is the
within-cell fraction.
"""
struct SampleInterpolant
    bin_idx::Vector{Int}
    t::Vector{Float64}
end

"""
    SampleInterpolant(samples, z_grid)

Precomputed interpolation metadata for fixed proposal redshifts on a shared
redshift grid. Stores the lower cell index and within-cell fraction for each
sample so likelihood evaluations can reuse bin locations.
"""
function SampleInterpolant(
        samples::AbstractVector{<:Real},
        z_grid::AbstractVector{<:Real}
)
    n_grid = length(z_grid)
    n_grid >= 2 || throw(ArgumentError("z_grid must contain at least two points"))
    n = length(samples)
    bin_idx = Vector{Int}(undef, n)
    t = Vector{Float64}(undef, n)
    z_min = @inbounds z_grid[1]
    z_max = @inbounds z_grid[end]
    @inbounds for i in 1:n
        z = samples[i]
        (z_min <= z <= z_max) || throw(
            ArgumentError("proposal redshift $(z) lies outside grid support [$z_min, $z_max]"),
        )
        idx = if z == z_max
            n_grid - 1
        else
            searchsortedlast(z_grid, z)
        end
        idx = max(1, min(idx, n_grid - 1))
        dz = z_grid[idx + 1] - z_grid[idx]
        bin_idx[i] = idx
        t[i] = Float64((z - z_grid[idx]) / dz)
    end
    return SampleInterpolant(bin_idx, t)
end

@inline function _interpolate_at_sample(
        y::AbstractVector,
        interp::SampleInterpolant,
        sample_index::Integer
)
    @inbounds begin
        i = interp.bin_idx[sample_index]
        t = interp.t[sample_index]
        return y[i] + t * (y[i + 1] - y[i])
    end
end

@inline function _cdf_at_sample(
        cumulative::AbstractVector,
        y::AbstractVector,
        interp::SampleInterpolant,
        z_grid::AbstractVector{<:Real},
        sample_index::Integer
)
    @inbounds begin
        i = interp.bin_idx[sample_index]
        t = interp.t[sample_index]
        dx = z_grid[i + 1] - z_grid[i]
        y_lo = y[i]
        y_hi = y[i + 1]
        return _linear_cell_integral(cumulative[i], y_lo, y_hi, dx, t)
    end
end

function build_redshift_prior(source_frame_fn, cache::CosmologyCache)
    z_grid_f = cache.inv_E_integral.x
    pdf_vals = map(eachindex(z_grid_f)) do i
        @inbounds z = z_grid_f[i]
        @inbounds d_c = cache.d_h * cache.inv_E_integral.cumulative[i]
        dvc_dz = cache.d_h * d_c^2 / E(z, cache.cosmology.Ωm)
        detector_frame_merger_rate_density(z, dvc_dz, source_frame_fn(z))
    end
    return RedshiftPrior(_cumulative_integral_from_values(z_grid_f, pdf_vals))
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

"""
    redshift_log_prob_samples!(out, prior, redshifts) -> out

Fill `out` with per-sample redshift log-density `_redshift_logpdf(prior, z)`.
`out` must have the same length as `redshifts`; values outside the prior's
support evaluate to `-Inf`.
"""
function redshift_log_prob_samples!(
        out::AbstractVector,
        prior::RedshiftPrior,
        redshifts::AbstractVector{<:Real}
)
    length(out) == length(redshifts) ||
        throw(ArgumentError("output length must match the number of redshift samples"))
    @inbounds for i in eachindex(redshifts)
        out[i] = _redshift_logpdf(prior, redshifts[i])
    end
    return out
end

"""
    redshift_log_prob_samples(prior, redshifts) -> Vector

Allocating variant of [`redshift_log_prob_samples!`](@ref). The element type
promotes with the redshift contribution (e.g. `ForwardDiff.Dual` when `prior`
was built under AD).
"""
function redshift_log_prob_samples(
        prior::RedshiftPrior,
        redshifts::AbstractVector{<:Real}
)
    T = promote_type(eltype(redshifts), redshift_logpdf_eltype(prior))
    out = Vector{T}(undef, length(redshifts))
    isempty(redshifts) && return out
    return redshift_log_prob_samples!(out, prior, redshifts)
end

function luminosity_distance_at_sample(
        cache::CosmologyCache,
        interp::SampleInterpolant,
        z_grid::AbstractVector{<:Real},
        z_samples::AbstractVector{<:Real},
        sample_index::Integer
)
    z = @inbounds z_samples[sample_index]
    integral = _cdf_at_sample(
        cache.inv_E_integral.cumulative,
        cache.inv_E_integral.y,
        interp,
        z_grid,
        sample_index
    )
    return (1 + z) * cache.d_h * integral
end

function build_redshift_prior(
        h::NamedTuple,
        spec::RedshiftPriorSpec,
        cache::CosmologyCache
)
    isnothing(spec.time_delay_model) || throw(
        ArgumentError("time-delay redshift models are not supported in the Julia v0 port"),
    )
    spec.family == MadauDickinson || throw(
        ArgumentError(
        "build_redshift_prior only supports the MadauDickinson redshift prior family",
    ),
    )
    # Streamlined closure: capture only the necessary scalars
    γ, κ, zpeak = h.γ, h.κ, h.zpeak
    sfn = z -> madau_dickinson_source_frame_distribution(z; γ, κ, zpeak)
    return build_redshift_prior(sfn, cache)
end

function cosmology_and_redshift_prior(
        h::NamedTuple,
        spec::RedshiftPriorSpec,
        z_grid::AbstractVector{<:Real} = redshift_grid(spec)
)
    cache = CosmologyCache(h, z_grid)
    return cache, build_redshift_prior(h, spec, cache)
end

function build_redshift_prior(
        h::NamedTuple,
        spec::RedshiftPriorSpec,
        z_grid::AbstractVector{<:Real} = redshift_grid(spec)
)
    return build_redshift_prior(h, spec, CosmologyCache(h, z_grid))
end


struct RedshiftInterpolatedDistribution{P <: RedshiftPrior} <:
       ContinuousUnivariateDistribution
    prior::P
end

Base.minimum(d::RedshiftInterpolatedDistribution) = first(d.prior.dN_dz.x)
Base.maximum(d::RedshiftInterpolatedDistribution) = last(d.prior.dN_dz.x)

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
