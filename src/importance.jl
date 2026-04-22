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
    d_l = luminosity_distance.(z, h.H0, h.Ωm, Ref(bundle.distance))
    dgw_theta = gravitational_wave_distance.(z, d_l, h.Ξ₀, h.Ξₙ)
    dgw_theta_sq = dgw_theta .^ 2

    prior = intrinsic_prior(problem.strategy, bundle)
    target_log_prob = intrinsic_log_prob_samples(prior, problem.proposal.samples)
    log_ratio = target_log_prob .- problem.proposal.log_prob
    weights = importance_weights(log_ratio, problem.proposal.dgw_fid_sq, dgw_theta_sq)
    return (;
        weights = weights,
        log_ratio = log_ratio,
        target_log_prob = target_log_prob,
        dgw_theta_sq = dgw_theta_sq
    )
end
