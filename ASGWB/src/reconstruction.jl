"""
    reconstruct_dgw_fid_sq(z, model, Λ) -> Vector{Float64}

Per-sample squared gravitational-wave luminosity distance at fiducial cosmology
and propagation, from source-frame redshifts `z`.
"""
function reconstruct_dgw_fid_sq(
        z::AbstractVector{<:Real},
        model::AbstractASGWBModel,
        Λ::NamedTuple
)::Vector{Float64}
    c = cosmology(model, Λ)
    d_l = luminosity_distance.(z, c)
    d_gw = gravitational_wave_distance.(model, z, d_l, Ref(Λ))
    return Float64.(d_gw .^ 2)
end

"""
    reconstruct_cached_flux_over_dgw2(cached_flux, z, model, Λ) -> Matrix{Float64}

Apply the squared ratio of electromagnetic to gravitational-wave luminosity distance
sample-wise. Inputs and outputs use the `(n_freq, n_samples)` layout (column-major
friendly; each proposal sample is a contiguous column).
"""
function reconstruct_cached_flux_over_dgw2(
        cached_flux::AbstractMatrix{<:Real},
        z::AbstractVector{<:Real},
        model::AbstractASGWBModel,
        Λ::NamedTuple
)::Matrix{Float64}
    size(cached_flux, 2) == length(z) ||
        throw(ArgumentError("cached_flux column count must match redshift sample count"))
    c = cosmology(model, Λ)
    d_l = luminosity_distance.(z, c)
    d_gw = gravitational_wave_distance.(model, z, d_l, Ref(Λ))
    scale_row = reshape(Float64.((d_l ./ d_gw) .^ 2), 1, :)
    return Matrix{Float64}(cached_flux) .* scale_row
end

"""
    reconstruct_proposal_log_prob(samples, spec, model, Λ) -> Vector{Float64}

Proposal log-density per sample: redshift grid log-density plus full-BNS intrinsic
uniform factors on [`FullBNSSamplesSoA`](@ref).
"""
function reconstruct_proposal_log_prob(
        samples::FullBNSSamplesSoA,
        spec::RedshiftPriorSpec,
        model::AbstractASGWBModel,
        Λ::NamedTuple
)::Vector{Float64}
    redshift_prior = build_redshift_prior(Λ, spec, cosmology(model, Λ))
    cached_log_prob = logpdf(intrinsic_prior(FullBNS()), samples)
    return cached_log_prob .+ redshift_log_prob_samples(redshift_prior, samples.redshift)
end
