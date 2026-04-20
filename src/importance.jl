using LinearAlgebra

"""
    importance_weights(log_ratio, dgw_fid_sq, dgw_theta_sq) -> Vector

Numerical importance weights: `exp(log_ratio) * dgw_fid_sq / dgw_theta_sq`. All inputs
are vectors of equal length; no high-level objects involved.
"""
function importance_weights(
    log_ratio::AbstractVector{<:Real},
    dgw_fid_sq::AbstractVector{<:Real},
    dgw_theta_sq::AbstractVector{<:Real},
)
    length(log_ratio) == length(dgw_fid_sq) == length(dgw_theta_sq) || throw(
        ArgumentError("importance weight inputs must have matching lengths"),
    )
    return exp.(log_ratio) .* dgw_fid_sq ./ dgw_theta_sq
end

"""
    compute_importance_weights(problem, h, bundle) -> NamedTuple

High-level builder: given the [`ImportanceSamplingProblem`](@ref), live
[`HyperParameters`](@ref), and a precomputed [`RedshiftGridBundle`](@ref), compute
per-sample importance weights and the intermediate quantities used by diagnostics
and the parity shim.

Returns a NamedTuple with fields `weights`, `log_ratio`, `target_log_prob`, `dgw_theta_sq`.
"""
function compute_importance_weights(
    problem::ImportanceSamplingProblem,
    h::HyperParameters,
    bundle::RadialInterpolant,
)
    z = redshift(problem)
    d_l = luminosity_distance.(z, h.cosmological.H0, h.cosmological.Omega_m, Ref(bundle.companion))
    dgw_theta = gravitational_wave_distance.(z, d_l, h.propagation.chi0, h.propagation.chin)
    dgw_theta_sq = dgw_theta .^ 2

    redshift_log_prob = log_prob_from_bundle.(z, Ref(bundle))
    target_log_prob = bns_intrinsic_log_prob_samples(problem.proposal.samples, redshift_log_prob)
    log_ratio = target_log_prob .- problem.proposal.log_prob
    weights = importance_weights(log_ratio, problem.proposal.dgw_fid_sq, dgw_theta_sq)
    return (;
        weights=weights,
        log_ratio=log_ratio,
        target_log_prob=target_log_prob,
        dgw_theta_sq=dgw_theta_sq,
    )
end
