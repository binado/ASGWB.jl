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

function SampleInterpolant(
        samples::AbstractVector{<:Real},
        z_grid::AbstractVector{<:Real}
)
    n_grid = length(z_grid)
    n_grid >= 2 || throw(ArgumentError("SampleInterpolant requires at least 2 grid points"))
    bin_idx = Vector{Int}(undef, length(samples))
    t = Vector{Float64}(undef, length(samples))
    z_min = @inbounds z_grid[1]
    z_max = @inbounds z_grid[end]
    last_bin = n_grid - 1

    @inbounds for i in eachindex(samples)
        z = samples[i]
        if z == z_max
            bin_idx[i] = last_bin
            t[i] = 1.0
            continue
        end
        if z < z_min || z > z_max
            throw(
                ArgumentError(
                "proposal redshift $(z) lies outside the interpolation support [$(z_min), $(z_max)]",
            ),
            )
        end
        idx = searchsortedlast(z_grid, z)
        1 <= idx < n_grid || throw(
            ArgumentError(
            "proposal redshift $(z) could not be assigned to an interpolation cell on [$(z_min), $(z_max)]",
        ),
        )
        z_lo = z_grid[idx]
        z_hi = z_grid[idx + 1]
        bin_idx[i] = idx
        t[i] = (z - z_lo) / (z_hi - z_lo)
    end

    return SampleInterpolant(bin_idx, t)
end

function interpolate_at_samples(
        y::AbstractVector,
        interp::SampleInterpolant,
        z_grid::AbstractVector{<:Real}
)
    length(y) == length(z_grid) ||
        throw(ArgumentError("y and z_grid must have matching lengths"))
    out = Vector{promote_type(eltype(y), Float64)}(undef, length(interp.bin_idx))
    @inbounds for i in eachindex(interp.bin_idx)
        idx = interp.bin_idx[i]
        y_lo = y[idx]
        out[i] = y_lo + interp.t[i] * (y[idx + 1] - y_lo)
    end
    return out
end

function cdf_at_samples(
        cumulative::AbstractVector,
        y::AbstractVector,
        interp::SampleInterpolant,
        z_grid::AbstractVector{<:Real}
)
    length(cumulative) == length(y) == length(z_grid) ||
        throw(ArgumentError("cumulative, y, and z_grid must have matching lengths"))
    out = Vector{promote_type(eltype(cumulative), eltype(y), Float64)}(
        undef,
        length(interp.bin_idx)
    )
    @inbounds for i in eachindex(interp.bin_idx)
        idx = interp.bin_idx[i]
        t = interp.t[i]
        y_lo = y[idx]
        y_hi = y[idx + 1]
        dx = z_grid[idx + 1] - z_grid[idx]
        out[i] = cumulative[idx] + dx * (y_lo * t + 0.5 * (y_hi - y_lo) * t^2)
    end
    return out
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
    inv_E_vals = map(w -> inv(E(w, Ωm)), z_grid_f)
    distance = _cumulative_integral_from_values(z_grid_f, inv_E_vals)
    d_h = SPEED_OF_LIGHT_KM_S / H0
    d_c = d_h .* distance.cumulative
    dvc_dz = d_h .* (d_c .^ 2) .* inv_E_vals
    source_frame_vals = map(source_frame_fn, z_grid_f)
    pdf_vals = detector_frame_merger_rate_density.(z_grid_f, dvc_dz, source_frame_vals)
    pdf = _cumulative_integral_from_values(z_grid_f, pdf_vals)
    return RedshiftBundle(distance, pdf)
end

function build_redshift_grid_bundle(
        h::HyperParametersNT,
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

function build_redshift_grid_bundle(h::HyperParametersNT, spec::RedshiftPriorSpec)
    return build_redshift_grid_bundle(h, spec, redshift_grid(spec))
end

function log_prob_from_bundle(value::Real, bundle::RedshiftBundle)
    norm = redshift_integral(bundle)
    T = promote_type(eltype(bundle.pdf.y), typeof(norm))
    tiny = floatmin(T)
    pdf_at_value = interpolate(bundle.pdf, value)
    return log(max(pdf_at_value / max(norm, tiny), tiny))
end

function log_prob_from_bundle(
        values::AbstractVector{<:Real},
        bundle::RedshiftBundle,
        interp::SampleInterpolant
)
    length(values) == length(interp.bin_idx) ||
        throw(ArgumentError("values and sample interpolant must have matching lengths"))
    pdf_at_values = interpolate_at_samples(bundle.pdf.y, interp, bundle.pdf.x)
    norm = redshift_integral(bundle)
    T = promote_type(eltype(pdf_at_values), typeof(norm))
    tiny = floatmin(T)
    norm_floor = max(norm, tiny)
    out = Vector{T}(undef, length(pdf_at_values))
    @inbounds for i in eachindex(pdf_at_values)
        out[i] = log(max(pdf_at_values[i] / norm_floor, tiny))
    end
    return out
end
