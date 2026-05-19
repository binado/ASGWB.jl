using Distributions
using Random

export RedshiftBundle, redshift_integral, merger_rate_per_sec,
       detector_frame_merger_rate_density, expected_number_of_events,
       madau_dickinson_source_frame_distribution, power_law_source_frame_distribution,
       redshift_grid, SampleInterpolant, _interpolate_at_sample, _cdf_at_sample,
       _build_redshift_grid, log_prob_from_bundle, luminosity_distance_at_sample,
       build_redshift_grid_bundle, RedshiftInterpolatedDistribution, _normalized_log_density

"""
    RedshiftBundle(distance, pdf)

Domain bundle pairing the comoving-distance cumulative integral with the
detector-frame merger-rate PDF cumulative integral, both sampled on the same
uniform redshift grid.

- `distance::CumulativeIntegral1D` : antiderivative of `1/E(z, Ωm)` used by
  [`comoving_distance`](@ref) / [`luminosity_distance`](@ref).
- `pdf::CumulativeIntegral1D`      : detector-frame merger-rate density; its
  [`normalizer`](@ref) is the redshift integral ``∫ p(z)\\,dz`` driving
  [`merger_rate_per_sec`](@ref) and its cumulative table supports inverse-CDF
  sampling in [`RedshiftInterpolatedDistribution`](@ref).
"""
struct RedshiftBundle{D <: CumulativeIntegral1D, P <: CumulativeIntegral1D}
    distance::D
    pdf::P
end

"""
    redshift_integral(bundle::RedshiftBundle) -> Real

Convenience wrapper for `normalizer(bundle.pdf)` — the detector-frame
redshift-integrated merger-rate density on the grid.
"""
redshift_integral(bundle::RedshiftBundle) = normalizer(bundle.pdf)

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
        bundle, local_merger_rate_gpc3_yr, observation_time_yr, observation_time_sec,
    ) -> Float64

Detector-frame merger rate in events/sec:
`expected_number_of_events(local_rate, redshift_integral(bundle), observation_time_yr) /
observation_time_sec`. `observation_time_yr` sets the events count, `observation_time_sec`
converts to per-second units; they are taken independently rather than assuming a fixed
seconds-per-year so the cache's stored pair of times round-trips exactly.
"""
function merger_rate_per_sec(
        bundle::RedshiftBundle,
        local_merger_rate_gpc3_yr::Real,
        observation_time_yr::Real,
        observation_time_sec::Real
)
    n_events = expected_number_of_events(
        local_merger_rate_gpc3_yr,
        redshift_integral(bundle),
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

function _build_redshift_grid(
        source_frame_fn,
        H0::Real,
        Ωm::Real,
        z_min::Real,
        z_max::Real,
        num_interp::Integer
)
    z_grid = collect(LinRange(Float64(z_min), Float64(z_max), Int(num_interp)))
    return _build_redshift_grid(source_frame_fn, H0, Ωm, z_grid)
end

function _build_redshift_grid(
        source_frame_fn,
        H0::Real,
        Ωm::Real,
        z_grid::AbstractVector{<:Real}
)
    z_grid_f = z_grid isa AbstractVector{Float64} ? z_grid : collect(Float64, z_grid)
    inv_E_vals = [inv(E(w, Ωm)) for w in z_grid_f]
    distance = _cumulative_integral_from_values(z_grid_f, inv_E_vals)
    d_h = SPEED_OF_LIGHT_KM_S / H0
    pdf_vals = map(eachindex(z_grid_f)) do i
        @inbounds z = z_grid_f[i]
        @inbounds d_c = d_h * distance.cumulative[i]
        dvc_dz = d_h * d_c^2 / E(z, Ωm)
        detector_frame_merger_rate_density(z, dvc_dz, source_frame_fn(z))
    end
    return RedshiftBundle(distance, _cumulative_integral_from_values(z_grid_f, pdf_vals))
end

@inline function _normalized_log_density(pdf_at_value, norm, tiny)
    return log(max(pdf_at_value / max(norm, tiny), tiny))
end

function log_prob_from_bundle(value::Real, bundle::RedshiftBundle)
    norm = redshift_integral(bundle)
    T = promote_type(eltype(bundle.pdf.y), typeof(norm))
    tiny = floatmin(T)
    pdf_at_value = interpolate(bundle.pdf, value)
    return _normalized_log_density(pdf_at_value, norm, tiny)
end

function luminosity_distance_at_sample(
        bundle::RedshiftBundle,
        H0::Real,
        interp::SampleInterpolant,
        z_grid::AbstractVector{<:Real},
        z_samples::AbstractVector{<:Real},
        sample_index::Integer
)
    z = @inbounds z_samples[sample_index]
    integral = _cdf_at_sample(
        bundle.distance.cumulative, bundle.distance.y, interp, z_grid, sample_index)
    return (1 + z) * (SPEED_OF_LIGHT_KM_S / H0) * integral
end

function build_redshift_grid_bundle(
        h::NamedTuple,
        spec::RedshiftPriorSpec,
        z_grid::AbstractVector{<:Real}
)
    isnothing(spec.time_delay_model) || throw(
        ArgumentError("time-delay redshift models are not supported in the Julia v0 port"),
    )
    spec.family == MadauDickinson || throw(
        ArgumentError(
        "build_redshift_grid_bundle only supports the MadauDickinson redshift prior family",
    ),
    )
    sfn = z -> madau_dickinson_source_frame_distribution(
        z;
        γ = h.γ,
        κ = h.κ,
        zpeak = h.zpeak
    )
    return _build_redshift_grid(
        sfn,
        h.H0,
        h.Ωm,
        z_grid
    )
end

function build_redshift_grid_bundle(h::NamedTuple, spec::RedshiftPriorSpec)
    return build_redshift_grid_bundle(h, spec, redshift_grid(spec))
end


struct RedshiftInterpolatedDistribution{B <: RedshiftBundle} <:
       ContinuousUnivariateDistribution
    bundle::B
end

Base.minimum(d::RedshiftInterpolatedDistribution) = first(d.bundle.pdf.x)
Base.maximum(d::RedshiftInterpolatedDistribution) = last(d.bundle.pdf.x)

function Distributions.insupport(d::RedshiftInterpolatedDistribution, value::Real)
    return minimum(d) <= value <= maximum(d)
end

function Distributions.logpdf(d::RedshiftInterpolatedDistribution, value::Real)
    insupport(d, value) || return -Inf
    return log_prob_from_bundle(value, d.bundle)
end

function Random.rand(rng::AbstractRNG, d::RedshiftInterpolatedDistribution)
    target = rand(rng) * redshift_integral(d.bundle)
    cumulative = d.bundle.pdf.cumulative
    x = d.bundle.pdf.x
    n = length(cumulative)
    idx = searchsortedlast(cumulative, target)
    idx <= 0 && return x[1]
    idx >= n && return x[end]
    c0, c1 = cumulative[idx], cumulative[idx + 1]
    x0, x1 = x[idx], x[idx + 1]
    c1 > c0 || return x0
    return x0 + (target - c0) * (x1 - x0) / (c1 - c0)
end
