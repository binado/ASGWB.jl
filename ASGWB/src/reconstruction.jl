"""
    reconstruct_dgw_fid_sq(z, fid::FiducialParameters) -> Vector{Float64}

Per-sample squared gravitational-wave luminosity distance at fiducial cosmology
and propagation (`fid`), from source-frame redshifts `z`.
"""
function reconstruct_dgw_fid_sq(
        z::AbstractVector{<:Real},
        fid::FiducialParameters
)::Vector{Float64}
    c = fiducial_cosmology(fid)
    d_l = luminosity_distance.(z, c)
    d_gw = gravitational_wave_distance.(z, d_l, fid.modified_gravity.Ξ₀, fid.modified_gravity.Ξₙ)
    return Float64.(d_gw .^ 2)
end

"""
    reconstruct_cached_flux_over_dgw2(cached_flux, z, fid::FiducialParameters) -> Matrix{Float64}

Apply the squared ratio of electromagnetic to gravitational-wave luminosity distance
sample-wise. Inputs and outputs use the `(n_freq, n_samples)` layout (column-major
friendly; each proposal sample is a contiguous column).
"""
function reconstruct_cached_flux_over_dgw2(
        cached_flux::AbstractMatrix{<:Real},
        z::AbstractVector{<:Real},
        fid::FiducialParameters
)::Matrix{Float64}
    size(cached_flux, 2) == length(z) ||
        throw(ArgumentError("cached_flux column count must match redshift sample count"))
    c = fiducial_cosmology(fid)
    d_l = luminosity_distance.(z, c)
    d_gw = gravitational_wave_distance.(z, d_l, fid.modified_gravity.Ξ₀, fid.modified_gravity.Ξₙ)
    scale_row = reshape(Float64.((d_l ./ d_gw) .^ 2), 1, :)
    return Matrix{Float64}(cached_flux) .* scale_row
end

"""
    reconstruct_proposal_log_prob(samples, spec, fid::FiducialParameters) -> Vector{Float64}

Proposal log-density per sample: redshift grid log-density plus full-BNS intrinsic
uniform factors on [`FullBNSSamplesSoA`](@ref).
"""
function reconstruct_proposal_log_prob(
        samples::FullBNSSamplesSoA,
        spec::RedshiftPriorSpec,
        fid::FiducialParameters
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
        fiducial_parameters::FiducialParameters;
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
