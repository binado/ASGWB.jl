using HDF5

const BUNDLE_COMMAND_ATTR = "command"
const BUNDLE_GIT_REVISION_ATTR = "git_revision"

"""
    load_problem(bundle_path, model_path, detectors; local_merger_rate, observation_time_yr) -> ImportanceSamplingProblem

Load a [`WaveformCatalog`](@ref) bundle and a `model.toml` file, verify
their model fingerprint matches, and build an in-memory
[`ImportanceSamplingProblem`](@ref) ready for MCMC.

`detectors` must contain at least two [`Detector`](@ref) values; the network effective
PSD and SGWB scale are computed from tabulated PSDs and overlap-reduction functions.

The fiducial spectral density stored in [`ObservationConfig`](@ref) is computed via
[`fiducial_spectral_density`](@ref) after the problem is assembled, so it always
reflects the current Julia forward model rather than any stale on-disk value.
"""
function load_problem(
        bundle_path::AbstractString,
        model_path::AbstractString,
        detectors::AbstractVector{D};
        local_merger_rate::Real,
        observation_time_yr::Real
)::ImportanceSamplingProblem where {D <: Detector}
    catalog = load_bundle(bundle_path)
    verify_model_fingerprint(catalog, model_path)
    config = load_model_config(model_path)
    return load_problem(
        catalog,
        config,
        detectors;
        local_merger_rate = local_merger_rate,
        observation_time_yr = observation_time_yr
    )
end

function load_problem(
        catalog::WaveformCatalog,
        config::ModelConfig,
        detectors::AbstractVector{D};
        local_merger_rate::Real,
        observation_time_yr::Real
)::ImportanceSamplingProblem where {D <: Detector}
    length(detectors) < 2 && throw(
        ArgumentError(
        "load_problem: at least two detectors are required to build effective_psd and sgwb_scale",
    ),
    )

    model = config.model
    require_redshift_population(model)
    Λ = config.fiducial_hyperparameters
    spec = config.redshift_prior_spec
    cache_C = cosmology_type(model)
    cache_C ∈ SUPPORTED_COSMOLOGIES || throw(
        ArgumentError(
        "model.toml specifies unsupported cosmology type $(cache_C); " *
        "supported: $(join(SUPPORTED_COSMOLOGIES, ", "))",
    ),
    )

    meta = catalog.metadata
    grid = meta.grid
    all_freq = frequencies(grid)
    mask = in_band_mask(grid)
    n_freq_full = length(all_freq)

    obs_yr = Float64(observation_time_yr)
    obs_sec = obs_yr * 365.25 * 24 * 3600.0

    det_vec = Vector{Detector}(collect(detectors))
    placeholder_fid_sd = zeros(Float64, n_freq_full)
    observation = build_observation_config(
        collect(Float64, all_freq),
        det_vec,
        mask,
        placeholder_fid_sd,
        obs_sec,
        obs_yr
    )

    z = catalog.samples.redshift
    n_samp = length(z)
    size(catalog.fluxes, 2) == n_samp || throw(
        ArgumentError(
        "bundle fluxes column count ($(size(catalog.fluxes, 2))) " *
        "does not match sample count ($n_samp)",
    ),
    )
    size(catalog.fluxes, 1) == n_freq_full || throw(
        ArgumentError(
        "bundle fluxes row count ($(size(catalog.fluxes, 1))) " *
        "does not match frequency grid length ($n_freq_full)",
    ),
    )

    intrinsic_site_order = _bns_intrinsic_site_order(catalog.samples)
    samples = _catalog_samples_to_bns_soa(catalog.samples)
    cached_flux_over_dgw2 = reconstruct_cached_flux_over_dgw2(catalog.fluxes, z, model, Λ)
    dgw_fid_sq = reconstruct_dgw_fid_sq(z, model, Λ)
    lp = reconstruct_proposal_log_prob(samples, spec, model, Λ)
    intrinsic_vector = _build_intrinsic_vector(catalog.samples, intrinsic_site_order)

    proposal = ProposalData(
        intrinsic_site_order,
        samples,
        lp,
        intrinsic_vector,
        cached_flux_over_dgw2,
        dgw_fid_sq
    )

    local_rate = Float64(local_merger_rate)

    redshift_cache = build_redshift_grid_cache(proposal, spec)
    strategy = resolve_intrinsic_strategy(proposal.intrinsic_site_order)
    p_shell = ImportanceSamplingProblem(
        proposal,
        observation,
        model,
        Λ,
        spec,
        redshift_cache.redshift_grid,
        redshift_cache,
        local_rate,
        strategy
    )

    fs = try
        fiducial_spectral_density(p_shell)
    catch err
        throw(
            ArgumentError(
            "fiducial_spectral_density recomputation failed after loading bundle; " *
            "Underlying error: " * sprint(showerror, err),
        ),
        )
    end
    observation2 = ObservationConfig(
        p_shell.observation.frequencies,
        p_shell.observation.effective_psd,
        p_shell.observation.sgwb_scale,
        p_shell.observation.in_band_mask,
        fs,
        p_shell.observation.observation_time_sec,
        p_shell.observation.observation_time_yr
    )
    return ImportanceSamplingProblem(
        proposal,
        observation2,
        model,
        Λ,
        spec,
        redshift_cache.redshift_grid,
        redshift_cache,
        local_rate,
        strategy
    )
end

const _BNS_INTRINSIC_KEYS = (
    :mass_1_source, :mass_2_source, :redshift,
    :chi_1, :chi_2, :lambda_1, :lambda_2
)

function _bns_intrinsic_site_order(samples::NamedTuple)
    for k in _BNS_INTRINSIC_KEYS
        haskey(samples, k) || throw(
            ArgumentError("bundle samples missing required BNS column $(repr(k))"),
        )
    end
    return [String(k) for k in _BNS_INTRINSIC_KEYS]
end

function _catalog_samples_to_bns_soa(samples::NamedTuple)::FullBNSSamplesSoA
    return (
        mass = stack_source_masses(samples.mass_1_source, samples.mass_2_source),
        redshift = copy(samples.redshift),
        χ₁ = copy(samples.chi_1),
        χ₂ = copy(samples.chi_2),
        Λ₁ = copy(samples.lambda_1),
        Λ₂ = copy(samples.lambda_2)
    )
end

function _build_intrinsic_vector(
        samples::NamedTuple,
        site_order::Vector{String}
)::Matrix{Float64}
    n = length(samples.redshift)
    ncols = length(site_order)
    mat = Matrix{Float64}(undef, n, ncols)
    for (j, key) in enumerate(site_order)
        k = Symbol(key)
        haskey(samples, k) || throw(ArgumentError("samples missing column $(repr(key))"))
        col = samples[k]
        for i in 1:n
            mat[i, j] = col[i]
        end
    end
    return mat
end
