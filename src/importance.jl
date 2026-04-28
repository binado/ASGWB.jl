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

@inline function _importance_terms_at_sample(
        problem::ImportanceSamplingProblem,
        h::HyperParametersNT,
        bundle::RedshiftBundle,
        norm::Real,
        tiny::Real,
        z::AbstractVector{<:Real},
        interp::SampleInterpolant,
        sample_index::Integer
)
    pdf_at_z = _interpolate_at_sample(bundle.pdf.y, interp, sample_index)
    redshift_log_prob = _normalized_log_density(pdf_at_z, norm, tiny)
    target_log_prob = problem.redshift_cache.intrinsic_log_prob_plan.fixed_log_prob[sample_index] +
                      redshift_log_prob
    log_ratio = target_log_prob - problem.proposal.log_prob[sample_index]
    d_l = luminosity_distance_at_sample(
        bundle,
        h.H0,
        interp,
        problem.redshift_cache.redshift_grid,
        z,
        sample_index
    )
    dgw_theta = gravitational_wave_distance(z[sample_index], d_l, h.Ξ₀, h.Ξₙ)
    dgw_theta_sq = dgw_theta^2
    weight = exp(log_ratio) * problem.proposal.dgw_fid_sq[sample_index] / dgw_theta_sq
    return target_log_prob, log_ratio, dgw_theta_sq, weight
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
    n = length(z)
    n == 0 && return (;
        weights = Float64[],
        log_ratio = Float64[],
        target_log_prob = Float64[],
        dgw_theta_sq = Float64[]
    )

    norm = redshift_integral(bundle)
    T = promote_type(eltype(bundle.pdf.y), typeof(norm))
    tiny = floatmin(T)
    interp = problem.redshift_cache.sample_interpolant
    first_terms = _importance_terms_at_sample(problem, h, bundle, norm, tiny, z, interp, 1)
    first_target, first_ratio, first_dgw_sq, first_weight = first_terms
    target_log_prob = Vector{typeof(first_target)}(undef, n)
    log_ratio = Vector{typeof(first_ratio)}(undef, n)
    dgw_theta_sq = Vector{typeof(first_dgw_sq)}(undef, n)
    weights = Vector{typeof(first_weight)}(undef, n)

    @inbounds begin
        target_log_prob[1] = first_target
        log_ratio[1] = first_ratio
        dgw_theta_sq[1] = first_dgw_sq
        weights[1] = first_weight
        for i in 2:n
            terms = _importance_terms_at_sample(
                problem, h, bundle, norm, tiny, z, interp, i)
            target, ratio, dgw_sq, weight = terms
            target_log_prob[i] = target
            log_ratio[i] = ratio
            dgw_theta_sq[i] = dgw_sq
            weights[i] = weight
        end
    end
    return (;
        weights = weights,
        log_ratio = log_ratio,
        target_log_prob = target_log_prob,
        dgw_theta_sq = dgw_theta_sq
    )
end
