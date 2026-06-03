"""
    merger_rate(problem, C, Λ, ctx::ModelContext) -> Float64

Detector-frame merger rate in events/sec at hyperparameters `Λ` (cosmology family `C`),
reading the local rate and observation times from `ctx`.
"""
function merger_rate(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        Λ::NamedTuple,
        ctx::ModelContext
) where {C <: AbstractCosmology}
    return merger_rate(
        problem,
        C,
        Λ,
        ctx.local_merger_rate,
        ctx.observation.observation_time_yr,
        ctx.observation.observation_time_sec
    )
end

"""
    merger_rate(problem, C, Λ, local_merger_rate, observation_time_yr, observation_time_sec) -> Float64

Detector-frame merger rate in events/sec from explicit observation arguments (no
`ModelContext` required). Doubles as the ctx-free oracle for the cached method above.
"""
function merger_rate(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        Λ::NamedTuple,
        local_merger_rate::Real,
        observation_time_yr::Real,
        observation_time_sec::Real
) where {C <: AbstractCosmology}
    c = cosmology(C, Λ)
    prior = single_event_prior(problem.population_model, c, Λ)
    return merger_rate(prior, local_merger_rate, observation_time_yr, observation_time_sec)
end

"""
    merger_rate(prior, local_merger_rate, observation_time_yr, observation_time_sec) -> Float64

Detector-frame merger rate in events/sec from an already-built `single_event_prior`. The
hot path constructs the prior once and shares it with `compute_importance_weights`, so this
is the form the forward model calls directly (no cosmology/prior rebuild). The redshift
`.dists.redshift.prior` reach-through stays here rather than at every call site.
"""
function merger_rate(
        prior::ProductNamedTupleDistribution,
        local_merger_rate::Real,
        observation_time_yr::Real,
        observation_time_sec::Real
)
    return merger_rate_per_sec(
        prior.dists.redshift.prior,
        local_merger_rate,
        observation_time_yr,
        observation_time_sec
    )
end

"""
    loglikelihood(Λ, problem, C, ctx; observed = ctx.fiducial_spectral_density)

Gaussian in-band log-likelihood of the SGWB spectral density at `Λ`. Inlines the
`weights → rate → Sₕ` sequence using the cached atomics and `ctx` masks/scales. Builds the
`single_event_prior` once and shares it between weights and rate (avoids rebuilding its
redshift `CosmologyCache` twice per ForwardDiff gradient step).
"""
function loglikelihood(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ctx::ModelContext;
        observed::AbstractVector{<:Real} = ctx.fiducial_spectral_density
) where {C <: AbstractCosmology}
    c = cosmology(C, Λ)
    prior = single_event_prior(problem.population_model, c, Λ)
    weights = compute_importance_weights(problem, c, prior, ctx)
    rate = merger_rate(
        prior,
        ctx.local_merger_rate,
        ctx.observation.observation_time_yr,
        ctx.observation.observation_time_sec
    )
    Sh = spectral_density(ctx.cached_flux_over_dgw2, rate; weights = weights)

    obs = ctx.observation
    mask = obs.in_band_mask
    σ = obs.sgwb_scale_in_band
    residual = observed[mask] .- Sh[mask]
    return -0.5 * sum((residual ./ σ) .^ 2 .+ log.(2π .* (σ .^ 2)))
end

function fiducial_hyperparameters(problem::ImportanceSamplingProblem)
    problem.fiducial_hyperparameters
end
