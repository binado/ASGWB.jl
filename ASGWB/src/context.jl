# --- private fiducial-cache reconstruction (relocated from reconstruction.jl) -------------
# `ModelContext` itself is defined in inference_types.jl so it precedes the importance.jl
# method signatures that dispatch on it.

# Per-sample squared GW luminosity distance at the fiducial cosmology/propagation.
function _reconstruct_dgw_fid_sq(
        z::AbstractVector{<:Real},
        ::Type{C},
        Λ::NamedTuple
)::Vector{Float64} where {C <: AbstractCosmology}
    c = cosmology(C, Λ)
    d_l = luminosity_distance.(z, c)
    d_gw = gravitational_wave_distance.(z, d_l, Ref(c))
    return Float64.(d_gw .^ 2)
end

# Apply the squared ratio (D_L / D_gw)² sample-wise to the raw catalog fluxes.
# Inputs/outputs use the `(n_freq, n_samples)` layout (each sample a contiguous column).
function _reconstruct_cached_flux_over_dgw2(
        fluxes::AbstractMatrix{<:Real},
        z::AbstractVector{<:Real},
        ::Type{C},
        Λ::NamedTuple
)::Matrix{Float64} where {C <: AbstractCosmology}
    size(fluxes, 2) == length(z) ||
        throw(ArgumentError("flux column count must match redshift sample count"))
    c = cosmology(C, Λ)
    d_l = luminosity_distance.(z, c)
    d_gw = gravitational_wave_distance.(z, d_l, Ref(c))
    scale_row = reshape(Float64.((d_l ./ d_gw) .^ 2), 1, :)
    return Matrix{Float64}(fluxes) .* scale_row
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

The proposal caches (`proposal_log_prob`, `dgw_fid_sq`, `cached_flux_over_dgw2`) are
recomputed from the raw catalog at the fiducial cosmology `cosmology(C, Λ_fid)`, so stale
on-disk values are never trusted. The `fiducial_spectral_density` is produced by running the
inline `weights → rate → Sₕ` sequence at the fiducial point.
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

    cached_flux_over_dgw2 = _reconstruct_cached_flux_over_dgw2(problem.fluxes, z, C, Λ_fid)
    dgw_fid_sq = _reconstruct_dgw_fid_sq(z, C, Λ_fid)
    proposal_log_prob = _reconstruct_proposal_log_prob(
        problem.samples, C, problem.population_model, Λ_fid)

    redshift_grid = collect(Float64, z_grid)
    interp = SampleInterpolant(z, redshift_grid)
    local_rate = Float64(local_merger_rate)

    # Fiducial spectral density: weights → rate → Sₕ at Λ_fid, through the same kernels the
    # likelihood uses, so the stored observed data matches the live forward model exactly.
    fs = try
        c_fid = cosmology(C, Λ_fid)
        cosmology_cache = CosmologyCache(c_fid, redshift_grid)
        prior_fid = single_event_prior(problem.population_model, c_fid, Λ_fid)
        weights_fid = _importance_weights_core(
            batched_logpdf(prior_fid, problem.samples),
            proposal_log_prob,
            dgw_fid_sq,
            z,
            redshift_grid,
            interp,
            cosmology_cache
        )
        rate_fid = merger_rate_per_sec(
            prior_fid.dists.redshift.prior, local_rate, obs_yr, obs_sec)
        spectral_density(cached_flux_over_dgw2, rate_fid; weights = weights_fid)
    catch err
        throw(
            ArgumentError(
            "fiducial_spectral_density computation failed while building model context; " *
            "underlying error: " * sprint(showerror, err),
        ),
        )
    end

    return ModelContext(
        proposal_log_prob,
        dgw_fid_sq,
        cached_flux_over_dgw2,
        redshift_grid,
        interp,
        observation,
        local_rate,
        fs
    )
end
