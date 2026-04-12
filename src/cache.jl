"""
    hyperparameters_from_fiducial(fid::ProposalFiducialParameters, spec::RedshiftPriorSpec) -> HyperParameters

Build [`HyperParameters`](@ref) from cache `hyperparameters` scalars and the file’s
[`RedshiftPriorSpec`](@ref). Used when reconstructing per-sample proposal log-density
from redshift grids (e.g. format v3 caches that omit `proposal_log_prob`).

Requires `gamma`, `kappa`, and `z_peak` on `fid` when `spec.family` is Madau–Dickinson,
and `lamb` when the family is power-law.
"""
function hyperparameters_from_fiducial(
    fid::ProposalFiducialParameters,
    spec::RedshiftPriorSpec,
)::HyperParameters
    if spec.family == MadauDickinson
        g, κ, zp = fid.gamma, fid.kappa, fid.z_peak
        if isnothing(g) || isnothing(κ) || isnothing(zp)
            throw(
                ArgumentError(
                    "reconstructing proposal log-density requires hyperparameters gamma, kappa, and z_peak for MadauDickinson redshift prior",
                ),
            )
        end
        return HyperParameters(;
            H0=fid.H0,
            Omega_m=fid.Omega_m,
            chi0=fid.chi0,
            chin=fid.chin,
            gamma=g,
            kappa=κ,
            z_peak=zp,
        )
    else
        λ = fid.lamb
        isnothing(λ) && throw(
            ArgumentError(
                "reconstructing proposal log-density requires hyperparameter lamb for PowerLaw redshift prior",
            ),
        )
        return HyperParameters(; H0=fid.H0, Omega_m=fid.Omega_m, chi0=fid.chi0, chin=fid.chin, lamb=λ)
    end
end

"""
    reconstruct_dgw_fid_sq(z, fid::ProposalFiducialParameters) -> Vector{Float64}

Per-sample squared gravitational-wave luminosity distance at fiducial cosmology
and propagation (`fid`), from source-frame redshifts `z`.
"""
function reconstruct_dgw_fid_sq(
    z::AbstractVector{<:Real},
    fid::ProposalFiducialParameters,
)::Vector{Float64}
    d_l = luminosity_distance.(z, fid.H0, fid.Omega_m)
    d_gw = gravitational_wave_distance.(z, d_l, fid.chi0, fid.chin)
    return Float64.(d_gw .^ 2)
end

"""
    reconstruct_cached_flux_over_dgw2(cached_flux, z, fid::ProposalFiducialParameters) -> Matrix{Float64}

Apply the squared ratio of electromagnetic to gravitational-wave luminosity
distance row-wise so the result matches the `cached_flux_over_dgw2` layout used
by [`ImportanceSamplingProblem`](@ref).
"""
function reconstruct_cached_flux_over_dgw2(
    cached_flux::AbstractMatrix{<:Real},
    z::AbstractVector{<:Real},
    fid::ProposalFiducialParameters,
)::Matrix{Float64}
    size(cached_flux, 1) == length(z) || throw(
        ArgumentError("cached_flux row count must match redshift sample count"),
    )
    d_l = luminosity_distance.(z, fid.H0, fid.Omega_m)
    d_gw = gravitational_wave_distance.(z, d_l, fid.chi0, fid.chin)
    scale_col = reshape(Float64.((d_l ./ d_gw) .^ 2), :, 1)
    return Matrix{Float64}(cached_flux) .* scale_col
end

"""
    reconstruct_proposal_log_prob(strategy, samples, spec, fid::ProposalFiducialParameters) -> Vector{Float64}

Proposal log-density per sample: redshift grid log-density from
[`hyperparameters_from_fiducial`](@ref) / [`build_redshift_grid_bundle`](@ref), plus
full-BNS intrinsic uniform factors when `strategy` is [`FullBNS`](@ref).
"""
function reconstruct_proposal_log_prob(
    strategy::IntrinsicPriorStrategy,
    samples::ProposalSampleBundle,
    spec::RedshiftPriorSpec,
    fid::ProposalFiducialParameters,
)::Vector{Float64}
    h = hyperparameters_from_fiducial(fid, spec)
    bundle = build_redshift_grid_bundle(h, spec)
    z = redshift(samples)
    rz = log_prob_from_bundle.(z, Ref(bundle))
    if strategy isa RedshiftOnly
        return Float64.(rz)
    elseif strategy isa FullBNS
        samples isa FullBNSSamples || throw(ArgumentError("FullBNS strategy requires FullBNSSamples"))
        return Float64.(bns_intrinsic_log_prob_samples(samples, rz))
    else
        throw(ArgumentError("unsupported intrinsic strategy for proposal log-density reconstruction"))
    end
end
