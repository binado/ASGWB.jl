using Distributions: logpdf, ProductNamedTupleDistribution

function target_log_prob_samples(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem
)
    model = problem.model
    c = cosmology(model, Λ)
    redshift_prior = build_redshift_prior(
        Λ,
        problem.redshift_prior_spec,
        c,
        problem.redshift_cache.redshift_grid
    )
    target_log_prob = problem.redshift_cache.cached_intrinsic_log_prob .+
                      redshift_log_prob_samples(
        redshift_prior,
        problem.proposal.samples.redshift
    )
    return target_log_prob, redshift_prior
end

"""
    evaluate_model_terms(model, Λ, problem, z_grid) -> NamedTuple

Evaluate the deterministic likelihood terms for a model hyperparameter state.
"""
function evaluate_model_terms(
        model::AbstractASGWBModel,
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem,
        z_grid::AbstractVector{<:Real}
)
    cosmology_cache,
    redshift_prior = cosmology_and_redshift_prior(
        cosmology(model, Λ),
        Λ,
        problem.redshift_prior_spec,
        z_grid
    )
    iw = compute_importance_weights(problem, Λ, cosmology_cache, redshift_prior)
    rate = merger_rate_per_sec(
        redshift_prior,
        problem.local_merger_rate,
        problem.observation.observation_time_yr,
        problem.observation.observation_time_sec
    )
    sd = spectral_density(problem.proposal.cached_flux_over_dgw2, rate; weights = iw.weights)
    return merge(iw,
        (
            redshift_integral = redshift_integral(redshift_prior),
            expected_number_of_sources = rate * problem.observation.observation_time_sec,
            spectral_density = sd,
            spectral_density_in_band = sd[problem.observation.in_band_mask]
        ))
end

"""
    evaluate_model_terms(model, Λ, problem) -> NamedTuple

Evaluate deterministic likelihood terms using the problem's redshift grid.
"""
function evaluate_model_terms(
        model::AbstractASGWBModel,
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem
)
    return evaluate_model_terms(model, Λ, problem, problem.redshift_cache.redshift_grid)
end

function evaluate_model_terms(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem
)
    return evaluate_model_terms(problem.model, Λ, problem)
end

function loglikelihood(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem;
        observed_spectral_density::AbstractVector{<:Real} = problem.observation.fiducial_spectral_density
)
    evaluation = evaluate_model_terms(Λ, problem)
    observed_in_band = observed_spectral_density[problem.observation.in_band_mask]
    residual = observed_in_band .- evaluation.spectral_density_in_band
    return -0.5 * sum(
        (residual ./ problem.observation.sgwb_scale_in_band) .^ 2 .+
        log.(2π .* (problem.observation.sgwb_scale_in_band .^ 2)),
    )
end

function logposterior(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution;
        observed_spectral_density::AbstractVector{<:Real} = problem.observation.fiducial_spectral_density
)
    return logpdf(prior, Λ) +
           loglikelihood(
        Λ,
        problem;
        observed_spectral_density = observed_spectral_density
    )
end

"""
    fiducial_hyperparameters(problem::ImportanceSamplingProblem) -> NamedTuple

Return the canonical fiducial hyperparameter state stored on `problem`.
"""
function fiducial_hyperparameters(problem::ImportanceSamplingProblem)
    problem.fiducial_hyperparameters
end

"""
    fiducial_spectral_density(problem::ImportanceSamplingProblem) -> Vector{Float64}

Strain spectral density ``S_h(f)`` at [`fiducial_hyperparameters`](@ref), from
[`evaluate_model_terms`](@ref) (cached flux, importance weights, and merger rate). Same
units as the vector returned by [`spectral_density`](@ref) on the importance-weighted fluxes.

For the dimensionless energy density ``\\Omega_{\\mathrm{GW}}(f)``, use [`Ωgw`](@ref) with
this vector and the corresponding frequency bins from `problem.observation.frequencies`.
"""
function fiducial_spectral_density(problem::ImportanceSamplingProblem)
    Λ = fiducial_hyperparameters(problem)
    return evaluate_model_terms(Λ, problem).spectral_density
end

"""
    fiducial_redshift_integral(model::AbstractASGWBModel, Λ, spec) -> Float64

[`cosmology_and_redshift_prior`](@ref) norm at the given hyperparameter state.
"""
function fiducial_redshift_integral(
        model::AbstractASGWBModel,
        Λ::NamedTuple,
        spec::RedshiftPriorSpec
)::Float64
    redshift_prior = build_redshift_prior(Λ, spec, cosmology(model, Λ))
    return Float64(redshift_integral(redshift_prior))
end

"""
    fiducial_redshift_integral(problem::ImportanceSamplingProblem) -> Float64

[`cosmology_and_redshift_prior`](@ref) norm at [`fiducial_hyperparameters`](@ref).
"""
function fiducial_redshift_integral(problem::ImportanceSamplingProblem)
    return fiducial_redshift_integral(
        problem.model,
        problem.fiducial_hyperparameters,
        problem.redshift_prior_spec
    )
end
