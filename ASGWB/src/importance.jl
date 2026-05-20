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
        cosmology_cache::CosmologyCache,
        prior::RedshiftPrior,
        norm::Real,
        tiny::Real,
        z::AbstractVector{<:Real},
        interp::SampleInterpolant,
        sample_index::Integer
)
    pdf_at_z = _interpolate_at_sample(prior.dN_dz.y, interp, sample_index)
    redshift_log_prob = _normalized_log_density(pdf_at_z, norm, tiny)
    target_log_prob = problem.redshift_cache.fixed_intrinsic_log_prob[sample_index] +
                      redshift_log_prob
    log_ratio = target_log_prob - problem.proposal.log_prob[sample_index]
    d_l = luminosity_distance_at_sample(
        cosmology_cache,
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

function _importance_output_eltypes(
        problem::ImportanceSamplingProblem,
        h::HyperParametersNT,
        cosmology_cache::CosmologyCache,
        prior::RedshiftPrior
)
    target_log_prob_type = promote_type(
        eltype(problem.redshift_cache.fixed_intrinsic_log_prob),
        redshift_logpdf_eltype(prior)
    )
    log_ratio_type = promote_type(target_log_prob_type, eltype(problem.proposal.log_prob))
    dgw_theta_sq_type = promote_type(
        eltype(redshift(problem)),
        typeof(h.H0),
        typeof(h.Ξ₀),
        typeof(h.Ξₙ),
        eltype(cosmology_cache.inv_E_integral.y),
        eltype(cosmology_cache.inv_E_integral.cumulative)
    )
    weight_type = promote_type(
        log_ratio_type,
        eltype(problem.proposal.dgw_fid_sq),
        dgw_theta_sq_type
    )
    return (;
        weights = weight_type,
        log_ratio = log_ratio_type,
        target_log_prob = target_log_prob_type,
        dgw_theta_sq = dgw_theta_sq_type
    )
end

"""
    compute_importance_weights(problem, h, cosmology_cache, prior) -> NamedTuple

High-level builder: given the [`ImportanceSamplingProblem`](@ref), live
[`HyperParameters`](@ref), a [`CosmologyCache`](@ref), and a [`RedshiftPrior`](@ref), compute
per-sample importance weights and the intermediate quantities used by diagnostics
and the parity shim.

Returns a NamedTuple with fields `weights`, `log_ratio`, `target_log_prob`, `dgw_theta_sq`.
"""
function compute_importance_weights(
        problem::ImportanceSamplingProblem,
        h::HyperParametersNT,
        cosmology_cache::CosmologyCache,
        prior::RedshiftPrior
)
    z = redshift(problem)
    n = length(z)
    norm = redshift_integral(prior)
    T = promote_type(eltype(prior.dN_dz.y), typeof(norm))
    tiny = floatmin(T)
    if n == 0
        eltypes = _importance_output_eltypes(problem, h, cosmology_cache, prior)
        return (;
            weights = Vector{eltypes.weights}(),
            log_ratio = Vector{eltypes.log_ratio}(),
            target_log_prob = Vector{eltypes.target_log_prob}(),
            dgw_theta_sq = Vector{eltypes.dgw_theta_sq}()
        )
    end

    interp = problem.redshift_cache.sample_interpolant
    first_terms = _importance_terms_at_sample(
        problem, h, cosmology_cache, prior, norm, tiny, z, interp, 1)
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
                problem, h, cosmology_cache, prior, norm, tiny, z, interp, i)
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
