using LinearAlgebra

"""
    importance_weights(log_ratio, dgw_fid_sq, dgw_theta_sq) -> Vector

Numerical importance weights: `exp(log_ratio) * dgw_fid_sq / dgw_theta_sq`. All inputs
are vectors of equal length; no high-level objects involved.
"""
function importance_weights(
        log_ratio::AbstractVector{<:Real},
        dgw_fid_sq::AbstractVector{<:Real},
        dgw_theta_sq::AbstractVector{<:Real}
)
    length(log_ratio) == length(dgw_fid_sq) == length(dgw_theta_sq) ||
        throw(ArgumentError("importance weight inputs must have matching lengths"))
    return exp.(log_ratio) .* dgw_fid_sq ./ dgw_theta_sq
end

"""
    compute_importance_weights(problem, h, bundle) -> NamedTuple

High-level builder: given the [`ImportanceSamplingProblem`](@ref), live
[`HyperParameters`](@ref), and a precomputed [`RedshiftBundle`](@ref), compute
per-sample importance weights and the intermediate quantities used by diagnostics
and the parity shim.

Returns a NamedTuple with fields `weights`, `log_ratio`, `target_log_prob`, `dgw_theta_sq`.
"""
function compute_importance_weights(
        problem::ImportanceSamplingProblem,
        h::HyperParametersNT,
        bundle::RedshiftBundle
)
    z = redshift(problem)
    d_c_over_d_h = cdf_at_samples(
        bundle.distance.cumulative,
        bundle.distance.y,
        problem.sample_interpolant,
        bundle.distance.x
    )
    d_h = SPEED_OF_LIGHT_KM_S / h.H0
    dgw_theta_sq = Vector{promote_type(
        eltype(d_c_over_d_h),
        typeof(d_h),
        typeof(h.Ξ₀),
        typeof(h.Ξₙ),
        eltype(z)
    )}(undef, length(z))
    @inbounds for i in eachindex(z)
        d_l = (1 + z[i]) * d_h * d_c_over_d_h[i]
        dgw_theta = gravitational_wave_distance(z[i], d_l, h.Ξ₀, h.Ξₙ)
        dgw_theta_sq[i] = dgw_theta^2
    end

    prior = intrinsic_prior(problem.strategy, bundle)
    target_log_prob = intrinsic_log_prob_samples(
        prior,
        problem.proposal.samples,
        problem.sample_interpolant
    )
    log_ratio = target_log_prob .- problem.proposal.log_prob
    weights = importance_weights(log_ratio, problem.proposal.dgw_fid_sq, dgw_theta_sq)
    return (;
        weights = weights,
        log_ratio = log_ratio,
        target_log_prob = target_log_prob,
        dgw_theta_sq = dgw_theta_sq
    )
end
