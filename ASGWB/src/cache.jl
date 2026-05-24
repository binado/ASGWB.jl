"""
    cosmology_type(fid::ProposalFiducialParameters) -> Type{<:AbstractCosmology}

Infer the cosmology subtype recorded on a cache fiducial (`w0` / `wa` optional fields).
"""
function cosmology_type(fid::ProposalFiducialParameters)
    if fid.wa !== nothing
        fid.w0 === nothing && throw(
            ArgumentError("cache provides wa without w0; cannot build W0WaCDM"),
        )
        return W0WaCDM
    elseif fid.w0 !== nothing
        return W0CDM
    else
        return LambdaCDM
    end
end

"""
    propagation_model(fid::ProposalFiducialParameters) -> MadauDickinsonModifiedPropagation

[`MadauDickinsonModifiedPropagation`](@ref) with cosmology type parameter matching `fid`.
"""
function propagation_model(fid::ProposalFiducialParameters)
    MadauDickinsonModifiedPropagation{cosmology_type(fid)}()
end

function _fiducial_cosmology_nt_fields(fid::ProposalFiducialParameters)
    C = cosmology_type(fid)
    base = (H0 = fid.H0, Ωm = fid.Ωm)
    C === LambdaCDM && return base
    C === W0CDM && return (; base..., w0 = fid.w0)
    return (; base..., w0 = fid.w0, wa = fid.wa)
end

"""
    fiducial_cosmology(fid::ProposalFiducialParameters) -> AbstractCosmology

Cosmology instance encoded on cache fiducial hyperparameters (`H0`, `Ωm`, optional `w0`/`wa`).
"""
function fiducial_cosmology(fid::ProposalFiducialParameters)
    return cosmology(cosmology_type(fid), _fiducial_cosmology_nt_fields(fid))
end

function _fiducial_cosmology_nt(fid::ProposalFiducialParameters)
    c = fiducial_cosmology(fid)
    fn = fieldnames(typeof(c))
    return NamedTuple{fn}(Tuple(getfield(c, f) for f in fn))
end

"""
    hyperparameters_from_fiducial(fid::ProposalFiducialParameters, spec::RedshiftPriorSpec) -> NamedTuple

Build model-validated fiducial hyperparameters from cache `hyperparameters` scalars and the file’s
[`RedshiftPriorSpec`](@ref). Used when reconstructing per-sample proposal log-density
from redshift grids (e.g. caches that omit `proposal_log_prob`).

Requires `γ`, `κ`, and `zpeak` on `fid` when `spec.family` is Madau–Dickinson,
Power-law caches are rejected: live hyperparameter reconstruction is MadauDickinson-only.
"""
function hyperparameters_from_fiducial(
        fid::ProposalFiducialParameters,
        spec::RedshiftPriorSpec
)
    spec.family == MadauDickinson || throw(
        ArgumentError(
        "live hyperparameter reconstruction supports MadauDickinson only; PowerLaw caches are metadata-only",
    ),
    )
    g, κ′, zp = fid.γ, fid.κ, fid.zpeak
    if isnothing(g) || isnothing(κ′) || isnothing(zp)
        throw(
            ArgumentError(
            "reconstructing proposal log-density requires hyperparameters gamma, kappa, and z_peak (HDF5 keys) for MadauDickinson redshift prior",
        ),
        )
    end
    model = propagation_model(fid)
    return canonical_hyperparameters(
        model,
        (;
            _fiducial_cosmology_nt(fid)...,
            Ξ₀ = fid.Ξ₀,
            Ξₙ = fid.Ξₙ,
            γ = g,
            κ = κ′,
            zpeak = zp
        );
        context = "fiducial hyperparameters"
    )
end

"""
    fiducial_redshift_integral(fid::ProposalFiducialParameters, spec::RedshiftPriorSpec) -> Float64

Trapezoid integral ``\\int p(z)\\,dz`` of the detector-frame merger-rate density on the
redshift grid defined by `spec` and the population in `hyperparameters_from_fiducial(fid, spec)`
(same population-key requirements as reconstructing an omitted `proposal_log_prob`).
"""
function fiducial_redshift_integral(
        fid::ProposalFiducialParameters,
        spec::RedshiftPriorSpec
)::Float64
    Λ = hyperparameters_from_fiducial(fid, spec)
    redshift_prior = build_redshift_prior(Λ, spec, cosmology(propagation_model(fid), Λ))
    return Float64(redshift_integral(redshift_prior))
end

"""
    reconstruct_dgw_fid_sq(z, fid::ProposalFiducialParameters) -> Vector{Float64}

Per-sample squared gravitational-wave luminosity distance at fiducial cosmology
and propagation (`fid`), from source-frame redshifts `z`.
"""
function reconstruct_dgw_fid_sq(
        z::AbstractVector{<:Real},
        fid::ProposalFiducialParameters
)::Vector{Float64}
    c = fiducial_cosmology(fid)
    d_l = luminosity_distance.(z, c)
    d_gw = gravitational_wave_distance.(z, d_l, fid.Ξ₀, fid.Ξₙ)
    return Float64.(d_gw .^ 2)
end

"""
    reconstruct_cached_flux_over_dgw2(cached_flux, z, fid::ProposalFiducialParameters) -> Matrix{Float64}

Apply the squared ratio of electromagnetic to gravitational-wave luminosity distance
sample-wise. Inputs and outputs use the `(n_freq, n_samples)` layout (column-major
friendly; each proposal sample is a contiguous column).
"""
function reconstruct_cached_flux_over_dgw2(
        cached_flux::AbstractMatrix{<:Real},
        z::AbstractVector{<:Real},
        fid::ProposalFiducialParameters
)::Matrix{Float64}
    size(cached_flux, 2) == length(z) ||
        throw(ArgumentError("cached_flux column count must match redshift sample count"))
    c = fiducial_cosmology(fid)
    d_l = luminosity_distance.(z, c)
    d_gw = gravitational_wave_distance.(z, d_l, fid.Ξ₀, fid.Ξₙ)
    scale_row = reshape(Float64.((d_l ./ d_gw) .^ 2), 1, :)
    return Matrix{Float64}(cached_flux) .* scale_row
end

"""
    reconstruct_proposal_log_prob(samples, spec, fid::ProposalFiducialParameters) -> Vector{Float64}

Proposal log-density per sample: redshift grid log-density from
[`hyperparameters_from_fiducial`](@ref) / [`cosmology_and_redshift_prior`](@ref), plus
full-BNS intrinsic uniform factors on [`FullBNSSamplesSoA`](@ref).
"""
function reconstruct_proposal_log_prob(
        samples::FullBNSSamplesSoA,
        spec::RedshiftPriorSpec,
        fid::ProposalFiducialParameters
)::Vector{Float64}
    Λ = hyperparameters_from_fiducial(fid, spec)
    redshift_prior = build_redshift_prior(Λ, spec, cosmology(propagation_model(fid), Λ))
    cached_log_prob = logpdf(intrinsic_prior(FullBNS()), samples)
    return cached_log_prob .+ redshift_log_prob_samples(redshift_prior, samples.redshift)
end

function importance_sampling_problem(
        proposal::ProposalData,
        observation::ObservationConfig,
        redshift_prior_spec::RedshiftPriorSpec,
        local_merger_rate::Real,
        fiducial_parameters::ProposalFiducialParameters;
        intrinsic_prior_factory = intrinsic_prior
)
    ri = fiducial_redshift_integral(fiducial_parameters, redshift_prior_spec)
    return importance_sampling_problem(
        proposal,
        observation,
        redshift_prior_spec,
        local_merger_rate,
        ri,
        fiducial_parameters;
        intrinsic_prior_factory = intrinsic_prior_factory
    )
end
