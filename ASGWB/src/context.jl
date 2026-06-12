# --- private fiducial-cache reconstruction (relocated from reconstruction.jl) -------------
# `ModelContext` itself is defined in inference_types.jl so it precedes the importance.jl
# method signatures that dispatch on it.

# Per-sample squared EM luminosity distance at the fiducial cosmology. The (Ξ₀, Ξₙ)
# propagation factor is *not* applied here: the full distance correction is carried live in
# the importance weights as `dl_fid_sq / (D_L,θ² · Ξ_θ²)`.
function _reconstruct_dl_fid_sq(
        z::AbstractVector{<:Real},
        ::Type{C},
        Λ::NamedTuple
)::Vector{Float64} where {C <: AbstractCosmology}
    c = cosmology(C, Λ)
    return Float64.(luminosity_distance.(z, c) .^ 2)
end

# Proposal log-density per sample: single-event prior logpdf at the fiducial point.
function _reconstruct_proposal_log_prob(
        samples::NamedTuple,
        ::Type{C},
        population::M,
        Λ::NamedTuple
)::Vector{Float64} where {C <: AbstractCosmology, M <: PopulationModel}
    c = cosmology(C, Λ)
    prior = single_event_prior(population, c, Λ)
    return batched_logpdf(prior, samples)
end

"""
    build_model_context(problem, C, grid, detectors, observation_time_yr, local_merger_rate; z_grid) -> ModelContext

Build the `Λ`-independent [`ModelContext`](@ref) for `problem` at cosmology family `C`.

`grid` is the catalog [`FrequencyGrid`](@ref) (supplies the frequency axis and analysis-band
mask). `detectors` must contain at least two [`Detector`](@ref)s; the network effective PSD
and Gaussian bin scales are computed from tabulated PSDs and overlap-reduction functions.

The proposal caches (`proposal_prior`, the per-component `proposal_log_prob`, `dl_fid_sq`)
are recomputed at the fiducial cosmology `cosmology(C, Λ_fid)`, so stale on-disk values are
never trusted. The raw catalog `fluxes` are used directly (no `(D_L/D_gw)²` pre-scaling).
The `fiducial_spectral_density` is produced by running the inline `weights → rate → Sₕ`
sequence at the fiducial point.
"""
function build_model_context(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        grid::FrequencyGrid,
        detectors::AbstractVector{D},
        observation_time_yr::Real,
        local_merger_rate::Real;
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
)::ModelContext where {C <: AbstractCosmology, D <: Detector}
    length(detectors) < 2 && throw(
        ArgumentError(
        "build_model_context: at least two detectors are required to build effective_psd and sgwb_scale",
    ),
    )
    C ∈ SUPPORTED_COSMOLOGIES || throw(
        ArgumentError(
        "unsupported cosmology type $(C); supported: $(join(SUPPORTED_COSMOLOGIES, ", "))",
    ),
    )

    Λ_fid = problem.fiducial_hyperparameters
    z = problem.samples.redshift
    n_samp = length(z)

    all_freq = frequencies(grid)
    mask = in_band_mask(grid)
    n_freq_full = length(all_freq)

    size(problem.fluxes, 2) == n_samp || throw(
        ArgumentError(
        "fluxes column count ($(size(problem.fluxes, 2))) does not match sample count ($n_samp)",
    ),
    )
    size(problem.fluxes, 1) == n_freq_full || throw(
        ArgumentError(
        "fluxes row count ($(size(problem.fluxes, 1))) does not match frequency grid length ($n_freq_full)",
    ),
    )

    obs_yr = Float64(observation_time_yr)
    obs_sec = obs_yr * 365.25 * 24 * 3600.0
    det_vec = Vector{Detector}(collect(detectors))
    observation = build_observation_context(all_freq, det_vec, mask, obs_sec, obs_yr)

    dl_fid_sq = _reconstruct_dl_fid_sq(z, C, Λ_fid)

    redshift_grid = collect(Float64, z_grid)
    interp = SampleInterpolant(z, redshift_grid)
    local_rate = Float64(local_merger_rate)

    # Fiducial proposal prior and its per-component log-densities, computed with the same
    # interpolant the hot path uses for the target redshift logpdf, so the redshift
    # log-ratio at Λ_fid is exactly zero (bitwise-identical arithmetic on both sides).
    cosmology_cache_fid = CosmologyCache(cosmology(C, Λ_fid), redshift_grid)
    proposal_prior = single_event_prior(
        problem.population_model, cosmology_cache_fid, Λ_fid)
    proposal_log_prob = component_logpdfs(
        proposal_prior, problem.samples, (; redshift = interp))

    # Fiducial spectral density: weights → rate → Sₕ at Λ_fid, through the same kernels the
    # likelihood uses, so the stored observed data matches the live forward model exactly.
    fs = try
        log_ratio_fid = logprobdiff(
            problem.population_model,
            proposal_prior,
            proposal_prior,
            proposal_log_prob,
            problem.samples,
            (; redshift = interp)
        )
        weights_fid = _importance_weights_core(
            log_ratio_fid,
            dl_fid_sq,
            z,
            redshift_grid,
            interp,
            cosmology_cache_fid
        )
        rate_fid = merger_rate_per_sec(
            proposal_prior.dists.redshift.prior, local_rate, obs_yr, obs_sec)
        spectral_density(problem.fluxes, rate_fid; weights = weights_fid)
    catch err
        throw(
            ArgumentError(
            "fiducial_spectral_density computation failed while building model context; " *
            "underlying error: " * sprint(showerror, err),
        ),
        )
    end

    return ModelContext(
        proposal_prior,
        proposal_log_prob,
        dl_fid_sq,
        redshift_grid,
        interp,
        observation,
        local_rate,
        fs
    )
end
