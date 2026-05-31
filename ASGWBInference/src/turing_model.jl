using Distributions: MvNormal, ProductNamedTupleDistribution
using LinearAlgebra: Diagonal
using Turing

function condition_turing_model(
        turing_model,
        theta0::NamedTuple,
        prior::ProductNamedTupleDistribution,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}};
        order::Tuple{Vararg{Symbol}}
)
    validate_hyperprior(order, prior)
    ordered_theta0 = canonical_hyperparameters(order, theta0; context = "initial hyperparameters")
    sample_only === nothing && return turing_model
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
    ),
    )
    validate_subset(sample_only, order)
    fixed = Tuple(s for s in order if s ∉ sample_only)
    isempty(fixed) && return turing_model
    return turing_model | (; (s => ordered_theta0[s] for s in fixed)...)
end

"""
    register_sample_hyperparameters(pop::PopulationModel)

Generate a `sample_hyperparameters` submodel for `pop` paired with each
supported cosmology.  `full_hyperparameters(C, pop)` is evaluated at
registration time so Turing sees literal symbol names in the generated
tilde-sites, which is what enables per-parameter conditioning via
[`condition_turing_model`](@ref).  Callers register their own population types.
"""
function register_sample_hyperparameters(pop::P) where {P <: PopulationModel}
    for C in SUPPORTED_COSMOLOGIES
        flds = full_hyperparameters(C, pop)
        @eval begin
            @model function sample_hyperparameters(c::Val{$C}, pop::$P, d)
                $([:($f ~ d.$f) for f in flds]...)
                return (; $(flds...))
            end
        end
    end
    return nothing
end

register_sample_hyperparameters(BNSPopulationModel())

@model function asgwb_importance_turing_model(
        track::Bool,
        problem::ImportanceSamplingProblem,
        ::Val{C},
        ctx::ModelContext,
        prior::ProductNamedTupleDistribution,
        observed_in_band::AbstractVector{<:Real}
) where {C}
    Λ ~ to_submodel(
        sample_hyperparameters(Val(C), problem.population_model, prior.dists), false)
    order = full_hyperparameters(C, problem.population_model)
    Λc = canonical_hyperparameters(
        order,
        Λ;
        context = "sampled hyperparameters",
        eltype = nothing
    )

    # Inline weights → rate → Sₕ from the cached atomics (R8): the generative model and the
    # ASGWBLogDensity likelihood write the same explicit sequence, divergence visible here.
    weights = compute_importance_weights(problem, C, Λc, ctx)
    rate = merger_rate(problem, C, Λc, ctx)
    Sh = spectral_density(ctx.cached_flux_over_dgw2, rate; weights = weights)

    obs = ctx.observation
    observed_in_band ~ MvNormal(
        Sh[obs.in_band_mask],
        Diagonal(obs.sgwb_scale_in_band .^ 2)
    )

    track || return nothing
    m = obs.in_band_mask
    df = frequency_bin_width(obs.frequencies)
    snr_sq = spectral_snr_squared(
        Sh[m], obs.effective_psd[m], obs.observation_time_sec, df)
    return (;
        number_of_sources = rate * obs.observation_time_sec,
        effective_sample_size = normalized_ess(weights),
        spectral_snr_squared = snr_sq,
        spectral_snr = sqrt(snr_sq)
    )
end

function build_turing_model(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ctx::ModelContext,
        prior::ProductNamedTupleDistribution;
        track::Bool = false,
        observed::AbstractVector{<:Real} = ctx.fiducial_spectral_density
) where {C <: AbstractCosmology}
    order = full_hyperparameters(C, problem.population_model)
    validate_hyperprior(order, prior)
    return asgwb_importance_turing_model(
        track,
        problem,
        Val(C),
        ctx,
        prior,
        observed[ctx.observation.in_band_mask]
    )
end
