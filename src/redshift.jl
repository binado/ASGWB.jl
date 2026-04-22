"""
    RedshiftBundle(distance, pdf)

Domain bundle pairing the comoving-distance cumulative integral with the
detector-frame merger-rate PDF cumulative integral, both sampled on the same
uniform redshift grid.

- `distance::CumulativeIntegral1D` : antiderivative of `1/E(z, Ω_m)` used by
  [`comoving_distance`](@ref) / [`luminosity_distance`](@ref).
- `pdf::CumulativeIntegral1D`      : detector-frame merger-rate density; its
  [`normalizer`](@ref) is the redshift integral ``∫ p(z)\\,dz`` driving
  [`merger_rate_per_sec`](@ref) and its cumulative table supports inverse-CDF
  sampling in [`RedshiftInterpolatedDistribution`](@ref).
"""
struct RedshiftBundle{D<:CumulativeIntegral1D, P<:CumulativeIntegral1D}
    distance::D
    pdf::P
end

"""
    redshift_integral(bundle::RedshiftBundle) -> Real

Convenience wrapper for `normalizer(bundle.pdf)` — the detector-frame
redshift-integrated merger-rate density on the grid.
"""
redshift_integral(bundle::RedshiftBundle) = normalizer(bundle.pdf)

function trapezoid_integral(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length"))
    length(x) >= 2 || throw(ArgumentError("at least two grid points are required"))
    @views sum((y[1:end-1] .+ y[2:end]) .* (x[2:end] .- x[1:end-1])) / 2
end

function detector_frame_merger_rate_density(
    z::Real,
    differential_comoving_volume::Real,
    source_frame_distribution::Real,
)
    return 4π * differential_comoving_volume * source_frame_distribution / (1 + z)
end

function expected_number_of_events(
    local_merger_rate_gpc3_yr::Real,
    redshift_integral_mpc3::Real,
    observation_time_yr::Real,
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
    observation_time_sec::Real,
)
    n_events = expected_number_of_events(
        local_merger_rate_gpc3_yr, redshift_integral(bundle), observation_time_yr,
    )
    return n_events / observation_time_sec
end

function madau_dickinson_source_frame_distribution(
    z::Real;
    gamma::Real,
    kappa::Real,
    z_peak::Real,
)
    one_plus_z = 1 + z
    return (
        (one_plus_z^gamma) / (1 + (one_plus_z / (1 + z_peak))^kappa)
    ) * (1 + (1 + z_peak)^(-kappa))
end

power_law_source_frame_distribution(z::Real; lamb::Real) = (1 + z)^lamb

function _build_redshift_grid(
    source_frame_fn,
    H0::Real,
    Omega_m::Real,
    z_min::Real,
    z_max::Real,
    num_interp::Integer,
)
    z_grid = collect(LinRange(Float64(z_min), Float64(z_max), Int(num_interp)))
    inv_E = w -> inv(E(w, Omega_m))
    distance = CumulativeIntegral1D(z_grid, inv_E)
    d_h = SPEED_OF_LIGHT_KM_S / H0
    pdf_integrand = let dist = distance, sf = source_frame_fn, Ω = Omega_m, dh = d_h
        function (w)
            d_c = dh * cdf(dist, w)
            dvc_dz = dh * d_c^2 / E(w, Ω)
            return detector_frame_merger_rate_density(w, dvc_dz, sf(w))
        end
    end
    pdf = CumulativeIntegral1D(z_grid, pdf_integrand)
    return RedshiftBundle(distance, pdf)
end

function build_redshift_grid_bundle(h::HyperParametersNT, spec::RedshiftPriorSpec)
    isnothing(spec.time_delay_model) || throw(
        ArgumentError("time-delay redshift models are not supported in the Julia v0 port"),
    )
    spec.family == MadauDickinson || throw(
        ArgumentError(
            "build_redshift_grid_bundle only supports the MadauDickinson redshift prior family",
        ),
    )
    sfn = z -> madau_dickinson_source_frame_distribution(
        z; gamma=h.gamma, kappa=h.kappa, z_peak=h.z_peak,
    )
    return _build_redshift_grid(sfn, h.H0, h.Omega_m, spec.z_min, spec.z_max, spec.num_interp)
end

function log_prob_from_bundle(value::Real, bundle::RedshiftBundle)
    norm = redshift_integral(bundle)
    T = promote_type(eltype(bundle.pdf.y), typeof(norm))
    tiny = floatmin(T)
    pdf_at_value = interpolate(bundle.pdf, value)
    return log(max(pdf_at_value / max(norm, tiny), tiny))
end
