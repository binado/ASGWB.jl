using Distributions: MvNormal, ProductNamedTupleDistribution
using LinearAlgebra: Diagonal
using Turing

function _require_supported_turing_model(model::MadauDickinsonModifiedPropagation)
    return nothing
end

function _require_supported_turing_model(model::AbstractASGWBModel)
    throw(
        ArgumentError(
        "Turing inference is implemented for MadauDickinsonModifiedPropagation; got $(typeof(model))",
    ),
    )
end

"""
    condition_turing_model(turing_model, theta0, prior, sample_only; model) -> model

If `sample_only === nothing`, return `model` unchanged (all hyperparameters are sampled).

Otherwise `sample_only` lists the subset of [`hyperparameters`](@ref)(`model`) that remain
stochastic; all other hyperparameters are **fixed** to the corresponding entries of `theta0`
using Turing's conditioning operator `|` (see
[Turing docs: conditioning on data](https://turinglang.org/docs/core-functionality/#conditioning-on-data)).

`model` must match the forward model used to build the Turing model (and the cache cosmology
for production runs).
"""
function condition_turing_model(
        turing_model,
        theta0::NamedTuple,
        prior::ProductNamedTupleDistribution,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}};
        model::AbstractASGWBModel
)
    _require_supported_turing_model(model)
    validate_prior(model, prior)
    ordered_theta0 = canonical_hyperparameters(model, theta0; context = "initial hyperparameters")
    sample_only === nothing && return turing_model
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
    ),
    )
    validate_subset(sample_only, model)
    order = hyperparameters(model)
    fixed = Tuple(s for s in order if s ∉ sample_only)
    isempty(fixed) && return turing_model
    return turing_model | (; (s => ordered_theta0[s] for s in fixed)...)
end

for C in SUPPORTED_COSMOLOGIES
    flds = cosmology_parameters(C)
    @eval begin
        @model function sample_cosmology(c::Val{$C}, d)
            $([:($f ~ d.$f) for f in flds]...)
            return (; $(flds...))
        end
    end
end

@model function sample_model_params(m::MadauDickinsonModifiedPropagation, d)
    Ξ₀ ~ d.Ξ₀
    Ξₙ ~ d.Ξₙ
    γ ~ d.γ
    κ ~ d.κ
    zpeak ~ d.zpeak
    return (; Ξ₀, Ξₙ, γ, κ, zpeak)
end

@model function asgwb_importance_turing_model(
        asgw_model::MadauDickinsonModifiedPropagation{C},
        track::Bool,
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution,
        observed_in_band::AbstractVector{<:Real}
) where {C <: AbstractCosmology}
    d = prior.dists
    cosmo_nt ~ to_submodel(sample_cosmology(Val(C), d), false)
    model_nt ~ to_submodel(sample_model_params(asgw_model, d), false)
    Λ = merge(cosmo_nt, model_nt)
    terms = evaluate_model_terms(
        asgw_model, Λ, problem, problem.redshift_cache.redshift_grid
    )

    observed_in_band ~ MvNormal(
        terms.spectral_density_in_band,
        Diagonal(problem.observation.sgwb_scale_in_band .^ 2)
    )

    track || return nothing
    obs = problem.observation
    m = obs.in_band_mask
    df = frequency_bin_width(obs.frequencies)
    snr_sq = spectral_snr_squared(
        terms.spectral_density[m], obs.effective_psd[m], obs.observation_time_sec, df
    )
    return (;
        number_of_sources = terms.expected_number_of_sources,
        effective_sample_size = normalized_ess(terms.weights),
        spectral_snr_squared = snr_sq,
        spectral_snr = sqrt(snr_sq)
    )
end

"""
    build_turing_model(problem, prior; model, track=false, observed_spectral_density=...) -> model

Construct a Turing `DynamicPPL.Model` for the ASGWB importance sampling likelihood.

# Arguments
- `problem::ImportanceSamplingProblem`: The pre-computed importance sampling cache.
- `prior::ProductNamedTupleDistribution`: Priors for the hyperparameters.
- `model::AbstractASGWBModel`: Forward model (must match cache cosmology for production).
- `track::Bool`: If `true`, the model returns a named tuple of diagnostic quantities
  (ESS, SNR, etc.) alongside the log-joint.
- `observed_spectral_density`: The "data" to condition on. Defaults to the fiducial
  spectral density from the cache.
"""
function build_turing_model(
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution;
        model::AbstractASGWBModel,
        track::Bool = false,
        observed_spectral_density::AbstractVector{<:Real} = problem.observation.fiducial_spectral_density
)
    _require_supported_turing_model(model)
    validate_prior(model, prior)
    return asgwb_importance_turing_model(
        model,
        track,
        problem,
        prior,
        observed_spectral_density[problem.observation.in_band_mask]
    )
end
