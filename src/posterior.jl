function target_log_prob_samples(h::HyperParametersNT, problem::ImportanceSamplingProblem)
    bundle = build_redshift_grid_bundle(h, problem.redshift_prior_spec)
    prior = intrinsic_prior(problem.strategy, bundle)
    return intrinsic_log_prob_samples(prior, problem.proposal.samples), bundle
end

"""
    evaluate_importance_terms(h, problem) -> NamedTuple

Thin composition of [`compute_importance_weights`](@ref), [`merger_rate_per_sec`](@ref),
and [`spectral_density`](@ref) that exposes every intermediate used by parity tests and
the AdvancedHMC likelihood (`dgw_theta_sq`, `target_log_prob`, `log_ratio`, `weights`,
`redshift_integral`, `expected_number_of_sources`, `spectral_density`,
`spectral_density_in_band`).
"""
function evaluate_importance_terms(h::HyperParametersNT, problem::ImportanceSamplingProblem)
    bundle = build_redshift_grid_bundle(h, problem.redshift_prior_spec)
    iw = compute_importance_weights(problem, h, bundle)
    rate = merger_rate_per_sec(
        bundle,
        problem.local_merger_rate,
        problem.observation.observation_time_yr,
        problem.observation.observation_time_sec,
    )
    sd = spectral_density(
        problem.proposal.cached_flux_over_dgw2, rate; weights=iw.weights,
    )
    return (
        dgw_theta_sq=iw.dgw_theta_sq,
        target_log_prob=iw.target_log_prob,
        log_ratio=iw.log_ratio,
        weights=iw.weights,
        redshift_integral=redshift_integral(bundle),
        expected_number_of_sources=rate * problem.observation.observation_time_sec,
        spectral_density=sd,
        spectral_density_in_band=sd[problem.observation.in_band_mask],
    )
end

function loglikelihood(
    h::HyperParametersNT,
    problem::ImportanceSamplingProblem;
    observed_spectral_density::AbstractVector{<:Real}=problem.observation.fiducial_spectral_density,
)
    evaluation = evaluate_importance_terms(h, problem)
    observed_in_band = observed_spectral_density[problem.observation.in_band_mask]
    residual = observed_in_band .- evaluation.spectral_density_in_band
    return -0.5 * sum(
        (residual ./ problem.observation.sgwb_scale_in_band) .^ 2 .+
        log.(2π .* (problem.observation.sgwb_scale_in_band .^ 2)),
    )
end

function logposterior(
    h::HyperParametersNT,
    problem::ImportanceSamplingProblem,
    prior::ProductNamedTupleDistribution;
    observed_spectral_density::AbstractVector{<:Real}=problem.observation.fiducial_spectral_density,
)
    return logprior(h, prior) + loglikelihood(
        h,
        problem;
        observed_spectral_density=observed_spectral_density,
    )
end

"""
    fiducial_hyperparameters(problem::ImportanceSamplingProblem) -> HyperParameters

Build [`HyperParameters`](@ref) from the cache’s [`ProposalFiducialParameters`](@ref)
and [`RedshiftPriorSpec`](@ref). Same rules as [`hyperparameters_from_fiducial`](@ref)
(population scalars on the proposal fiducial dict when the prior family requires them).
"""
function fiducial_hyperparameters(problem::ImportanceSamplingProblem)
    return hyperparameters_from_fiducial(
        problem.fiducial_parameters,
        problem.redshift_prior_spec,
    )
end

"""
    fiducial_spectral_density(problem::ImportanceSamplingProblem) -> Vector{Float64}

Strain spectral density ``S_h(f)`` at [`fiducial_hyperparameters`](@ref), from
[`evaluate_importance_terms`](@ref) (cached flux, importance weights, and merger rate). Same
units as the vector returned by [`spectral_density`](@ref) on the importance-weighted fluxes.

For the dimensionless energy density ``\\Omega_{\\mathrm{GW}}(f)``, use [`omegagw`](@ref) with
this vector and the corresponding frequency bins from `problem.observation.frequencies`.
"""
function fiducial_spectral_density(problem::ImportanceSamplingProblem)
    h = fiducial_hyperparameters(problem)
    return evaluate_importance_terms(h, problem).spectral_density
end

"""
    fiducial_redshift_integral(problem::ImportanceSamplingProblem) -> Float64

[`build_redshift_grid_bundle`](@ref) norm at [`fiducial_hyperparameters`](@ref) (matches
`problem.redshift_integral_fiducial` when that field was set from the fiducial population).
"""
function fiducial_redshift_integral(problem::ImportanceSamplingProblem)
    return fiducial_redshift_integral(problem.fiducial_parameters, problem.redshift_prior_spec)
end
