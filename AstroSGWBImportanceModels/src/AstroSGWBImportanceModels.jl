"""
    AstroSGWBImportanceModels

Concrete, reusable importance-model adapters for `AstroSGWBInference`. The package owns
astrophysical model choices while `AstroSGWBInference` retains the generic two-method
model contract.
"""
module AstroSGWBImportanceModels

import AstroSGWBInference: hyperparameters, merger_rate_and_log_weights
using CBCDistributions:
                        DEFAULT_Z_GRID,
                        GridQuery,
                        MadauDickinsonSourceFrame,
                        _normalized_log_density,
                        build_redshift_prior,
                        interpolate,
                        merger_rate_per_sec,
                        redshift_integral,
                        redshift_logpdf_eltype,
                        source_frame_distribution
import Cosmology
using Cosmology:
                 AbstractCosmology,
                 AbstractPropagation,
                 CosmologyCache,
                 cosmology,
                 gw_em_distance_ratio,
                 luminosity_distance,
                 luminosity_distance_at_sample,
                 propagation

export BNSMadauDickinsonImportanceModel,
       bns_madau_dickinson_hyperparameters,
       bns_samples_from_catalog,
       prepare_bns_madau_dickinson_model

"""
    BNSMadauDickinsonImportanceModel{C, P}

Prepared BNS importance model using a Madau–Dickinson source-frame merger rate,
background cosmology `C`, and GW propagation model `P`. Detector state is intentionally
kept in a separate `AstroSGWB.ObservationContext`.
"""
struct BNSMadauDickinsonImportanceModel{
    C <: AbstractCosmology, P <: AbstractPropagation}
    z_grid::Vector{Float64}
    query::GridQuery
    proposal_log_pdf::Vector{Float64}
    local_merger_rate::Float64
    observation_time::Float64
end

"""
    bns_madau_dickinson_hyperparameters(C, P) -> Tuple{Vararg{Symbol}}

Return the cosmology, propagation, and Madau–Dickinson hyperparameter names for type
tokens `C` and `P`.
"""
function bns_madau_dickinson_hyperparameters(
        ::Type{C}, ::Type{P}) where {C <: AbstractCosmology, P <: AbstractPropagation}
    return (Cosmology.hyperparameters(C)...,
        Cosmology.propagation_hyperparameters(P)..., :γ, :κ, :zpeak)
end

function hyperparameters(
        ::BNSMadauDickinsonImportanceModel{
        C, P}) where {
        C <: AbstractCosmology, P <: AbstractPropagation}
    return bns_madau_dickinson_hyperparameters(C, P)
end

"""
    bns_samples_from_catalog(catalog_samples, C, fiducials) -> NamedTuple

Keep the catalog columns used by the BNS importance-weight loop. A stored
`luminosity_distance` column is copied verbatim; otherwise EM luminosity distances are
synthesized at the fiducial cosmology `C`.
"""
function bns_samples_from_catalog(
        catalog_samples::NamedTuple,
        ::Type{C},
        fiducials::NamedTuple
) where {C <: AbstractCosmology}
    z = copy(catalog_samples.redshift)
    d_l = haskey(catalog_samples, :luminosity_distance) ?
          copy(catalog_samples.luminosity_distance) :
          luminosity_distance.(z, Ref(cosmology(C, fiducials)))
    return (redshift = z, luminosity_distance = d_l)
end

"""
    prepare_bns_madau_dickinson_model(samples, fiducials, C, P;
        local_merger_rate, observation_time, z_grid=DEFAULT_Z_GRID)

Precompute the Float64 proposal caches for the canonical BNS Madau–Dickinson importance
adapter. `local_merger_rate` is the local merger rate in events per year; `observation_time`
is the observation duration in years (Julian year). Returns the prepared model directly.
Construct detector state separately with `AstroSGWB.build_observation_context`.
"""
function prepare_bns_madau_dickinson_model(
        samples::NamedTuple,
        fiducials::NamedTuple,
        ::Type{C},
        ::Type{P};
        local_merger_rate::Real,
        observation_time::Real,
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
) where {C <: AbstractCosmology, P <: AbstractPropagation}
    z = samples.redshift
    zg = collect(Float64, z_grid)
    query = GridQuery(z, zg)
    prior_fid = build_redshift_prior(
        zz -> source_frame_distribution(MadauDickinsonSourceFrame(), zz, fiducials),
        CosmologyCache(cosmology(C, fiducials), zg))
    norm_fid = redshift_integral(prior_fid)
    tiny = floatmin(Float64)
    proposal_log_pdf = [_normalized_log_density(
                            interpolate(prior_fid.dN_dz, query, i), norm_fid, tiny)
                        for i in eachindex(z)]

    return BNSMadauDickinsonImportanceModel{C, P}(
        zg,
        query,
        proposal_log_pdf,
        Float64(local_merger_rate),
        Float64(observation_time)
    )
end

function merger_rate_and_log_weights(
        model::BNSMadauDickinsonImportanceModel{C, P},
        Λ::NamedTuple,
        samples
) where {C <: AbstractCosmology, P <: AbstractPropagation}
    z = samples.redshift
    d_l_fid = samples.luminosity_distance
    cache = CosmologyCache(cosmology(C, Λ), model.z_grid)
    prop = propagation(P, Λ)

    prior = build_redshift_prior(
        zz -> source_frame_distribution(MadauDickinsonSourceFrame(), zz, Λ), cache)
    norm = redshift_integral(prior)
    tiny = floatmin(real(eltype(prior.dN_dz.y)))

    T = promote_type(redshift_logpdf_eltype(prior),
        typeof(gw_em_distance_ratio(zero(eltype(z)), prop)))
    log_weights = Vector{T}(undef, length(z))
    @inbounds for i in eachindex(z)
        log_p_target = _normalized_log_density(
            interpolate(prior.dN_dz, model.query, i), norm, tiny)
        d_l_θ = luminosity_distance_at_sample(cache, model.query, z, i)
        Ξ_θ = gw_em_distance_ratio(z[i], prop)
        log_weights[i] = (log_p_target - model.proposal_log_pdf[i]) +
                         2 * log(d_l_fid[i]) - 2 * log(d_l_θ) - 2 * log(Ξ_θ)
    end

    rate = merger_rate_per_sec(prior, model.local_merger_rate, model.observation_time)
    return (rate, log_weights)
end

end
