"""
    hyperparameters_from_fiducial(fid::ProposalFiducialParameters, spec::RedshiftPriorSpec) -> HyperParameters

Build [`HyperParameters`](@ref) from cache `hyperparameters` scalars and the file’s
[`RedshiftPriorSpec`](@ref). Used when reconstructing per-sample proposal log-density
from redshift grids (e.g. caches that omit `proposal_log_prob`).

Requires `γ`, `κ`, and `zpeak` on `fid` when `spec.family` is Madau–Dickinson,
Power-law caches are rejected: live hyperparameter reconstruction is MadauDickinson-only.
"""
function hyperparameters_from_fiducial(
        fid::ProposalFiducialParameters,
        spec::RedshiftPriorSpec
)::HyperParameters
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
    return HyperParameters(;
        H0 = fid.H0,
        Ωm = fid.Ωm,
        Ξ₀ = fid.Ξ₀,
        Ξₙ = fid.Ξₙ,
        γ = g,
        κ = κ′,
        zpeak = zp
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
    h = hyperparameters_from_fiducial(fid, spec)
    redshift_prior = build_redshift_prior(h, spec)
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
    d_l = luminosity_distance.(z, fid.H0, fid.Ωm)
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
    d_l = luminosity_distance.(z, fid.H0, fid.Ωm)
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
    h = hyperparameters_from_fiducial(fid, spec)
    redshift_prior = build_redshift_prior(h, spec)
    fixed_log_prob = fixed_intrinsic_log_prob(FullBNS(), samples)
    return fixed_log_prob .+ redshift_log_prob_samples(redshift_prior, samples.redshift)
end

function importance_sampling_problem(
        proposal::ProposalData,
        observation::ObservationConfig,
        redshift_prior_spec::RedshiftPriorSpec,
        local_merger_rate::Real,
        fiducial_parameters::ProposalFiducialParameters
)
    ri = fiducial_redshift_integral(fiducial_parameters, redshift_prior_spec)
    return importance_sampling_problem(
        proposal,
        observation,
        redshift_prior_spec,
        local_merger_rate,
        ri,
        fiducial_parameters
    )
end
