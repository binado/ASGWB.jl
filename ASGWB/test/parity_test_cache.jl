# Test-only synthetic bundle fixtures. Include after `using ASGWB` (see `runtests.jl`).

const _PARITY_COMMAND = "ASGWB/test/parity_test_cache.jl (generated test bundle)"
const _PARITY_GIT_REVISION = "parity-snapshots"
const _PARITY_FREQUENCY_GRID = FrequencyGrid(0.05, 80.0, 20.0, 15.0, 45.0)

function _parity_model_toml_content(; H0 = 67.0, Omega_m = 0.315, Xi_0 = 1.0, Xi_n = 0.0,
        gamma = 2.7, kappa = 3.0, z_peak = 2.5,
        z_min = 0.001, z_max = 20.0, num_interp = 64)
    return """
[model]
cosmology = "ModifiedPropagation{LambdaCDM}"

[parameters]
H0 = $H0
Omega_m = $Omega_m
Xi_0 = $Xi_0
Xi_n = $Xi_n
gamma = $gamma
kappa = $kappa
z_peak = $z_peak

[population.redshift]
model = "madau_dickinson_source_frame"
z_min = $z_min
z_max = $z_max
num_interp = $num_interp
time_delay_model = "none"
"""
end

function _parity_model_toml_content_w0(; H0 = 67.0, Omega_m = 0.315, w0 = -0.9,
        Xi_0 = 1.0, Xi_n = 0.0,
        gamma = 2.7, kappa = 3.0, z_peak = 2.5,
        z_min = 0.001, z_max = 20.0, num_interp = 64)
    return """
[model]
cosmology = "ModifiedPropagation{W0CDM}"

[parameters]
H0 = $H0
Omega_m = $Omega_m
w0 = $w0
Xi_0 = $Xi_0
Xi_n = $Xi_n
gamma = $gamma
kappa = $kappa
z_peak = $z_peak

[population.redshift]
model = "madau_dickinson_source_frame"
z_min = $z_min
z_max = $z_max
num_interp = $num_interp
time_delay_model = "none"
"""
end

function _write_parity_bundle!(dir::String, variant::Symbol)
    if variant == :posterior
        _write_posterior_bundle(dir)
    elseif variant == :full_intrinsic
        _write_full_intrinsic_bundle(dir)
    elseif variant == :importance_context || variant == :posterior_v2_minimal
        _write_importance_context_bundle(dir)
    elseif variant == :w0cdm
        _write_w0cdm_bundle(dir)
    else
        throw(ArgumentError("unknown parity bundle variant $(repr(variant))"))
    end
    return dir
end

function _write_model_toml(dir, content)
    path = joinpath(dir, "model.toml")
    write(path, content)
    return path
end

function _write_bundle_h5(dir, catalog)
    path = joinpath(dir, "bundle.h5")
    save_bundle(path, catalog)
    return path
end

function _make_bns_samples(masses1, masses2, redshifts; chi1 = nothing, chi2 = nothing,
        lambda1 = nothing, lambda2 = nothing, luminosity_distances = nothing)
    n = length(redshifts)
    chi1 = isnothing(chi1) ? fill(0.0, n) : chi1
    chi2 = isnothing(chi2) ? fill(0.0, n) : chi2
    lambda1 = isnothing(lambda1) ? fill(100.0, n) : lambda1
    lambda2 = isnothing(lambda2) ? fill(100.0, n) : lambda2
    luminosity_distances = isnothing(luminosity_distances) ?
                           fill(500.0, n) : luminosity_distances
    return (
        mass_1_source = collect(Float64, masses1),
        mass_2_source = collect(Float64, masses2),
        redshift = collect(Float64, redshifts),
        chi_1 = collect(Float64, chi1),
        chi_2 = collect(Float64, chi2),
        lambda_1 = collect(Float64, lambda1),
        lambda_2 = collect(Float64, lambda2),
        luminosity_distance = collect(Float64, luminosity_distances)
    )
end

function _write_posterior_bundle(dir)
    toml_content = _parity_model_toml_content(;
        gamma = 2.7, kappa = 5.7, z_peak = 2.0, num_interp = 64)
    model_path = _write_model_toml(dir, toml_content)
    sha = model_sha256_of_file(model_path)

    samples = _make_bns_samples(
        [1.4, 1.4], [1.2, 1.2], [0.1, 0.2];
        luminosity_distances = [430.0, 880.0]
    )
    grid = _PARITY_FREQUENCY_GRID
    cached_flux = Float64[0.0 0.0; 1.0 4.0; 2.0 5.0]
    metadata = WaveformMetadata(
        "IMRPhenomPV2_NRTidalv2", :BNS, grid, sha,
        _PARITY_GIT_REVISION, _PARITY_COMMAND
    )
    catalog = WaveformCatalog(samples, cached_flux, metadata)
    _write_bundle_h5(dir, catalog)
    return dir
end

function _write_full_intrinsic_bundle(dir)
    toml_content = _parity_model_toml_content(;
        gamma = 2.7, kappa = 5.7, z_peak = 2.0, num_interp = 64)
    model_path = _write_model_toml(dir, toml_content)
    sha = model_sha256_of_file(model_path)

    samples = _make_bns_samples(
        [1.8, 2.2, 1.4, 2.4], [1.2, 1.7, 1.1, 1.3], [0.1, 0.2, 0.3, 0.5];
        chi1 = [0.0, 0.2, -0.1, 0.5],
        chi2 = [0.1, -0.2, 0.0, 0.3],
        lambda1 = [400.0, 800.0, 1200.0, 2000.0],
        lambda2 = [300.0, 600.0, 700.0, 1500.0],
        luminosity_distances = [430.0, 880.0, 1350.0, 2300.0]
    )
    grid = _PARITY_FREQUENCY_GRID
    cached_flux = Float64[0.0 0.0 0.0 0.0
                          1.0 1.5 2.0 2.5
                          2.0 2.5 3.0 3.5]
    metadata = WaveformMetadata(
        "IMRPhenomPV2_NRTidalv2", :BNS, grid, sha,
        _PARITY_GIT_REVISION, _PARITY_COMMAND
    )
    catalog = WaveformCatalog(samples, cached_flux, metadata)
    _write_bundle_h5(dir, catalog)
    return dir
end

function _write_importance_context_bundle(dir)
    toml_content = _parity_model_toml_content(;
        gamma = 2.7, kappa = 3.0, z_peak = 2.5, num_interp = 1024)
    model_path = _write_model_toml(dir, toml_content)
    sha = model_sha256_of_file(model_path)

    samples = _make_bns_samples(
        [1.4, 1.4], [1.2, 1.2], [0.1, 0.2];
        luminosity_distances = [430.0, 880.0]
    )
    grid = _PARITY_FREQUENCY_GRID
    cached_flux = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    metadata = WaveformMetadata(
        "IMRPhenomPV2_NRTidalv2", :BNS, grid, sha,
        _PARITY_GIT_REVISION, _PARITY_COMMAND
    )
    catalog = WaveformCatalog(samples, cached_flux, metadata)
    _write_bundle_h5(dir, catalog)
    return dir
end

function _write_w0cdm_bundle(dir)
    toml_content = _parity_model_toml_content_w0(;
        gamma = 2.7, kappa = 3.0, z_peak = 2.5, num_interp = 64)
    model_path = _write_model_toml(dir, toml_content)
    sha = model_sha256_of_file(model_path)

    samples = _make_bns_samples(
        [1.4, 1.4], [1.2, 1.2], [0.1, 0.2];
        luminosity_distances = [430.0, 880.0]
    )
    grid = _PARITY_FREQUENCY_GRID
    cached_flux = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    metadata = WaveformMetadata(
        "IMRPhenomPV2_NRTidalv2", :BNS, grid, sha,
        _PARITY_GIT_REVISION, _PARITY_COMMAND
    )
    catalog = WaveformCatalog(samples, cached_flux, metadata)
    _write_bundle_h5(dir, catalog)
    return dir
end

const _PARITY_BUNDLE_DIRS = Dict{Symbol, String}()

function parity_observation_kwargs(variant::Symbol)
    if variant == :posterior || variant == :full_intrinsic
        return (local_merger_rate = 1e-7, observation_time_yr = 1e-6)
    else
        return (local_merger_rate = 161.0, observation_time_yr = 1.0)
    end
end

"""
    parity_bundle_dir(variant) -> String

Return the directory containing `model.toml` and `bundle.h5` for `variant`.
The bundle is generated lazily on first call.

Variants: `:posterior`, `:full_intrinsic`, `:importance_context`,
`:posterior_v2_minimal` (alias for `:importance_context`), `:w0cdm`.
"""
function parity_bundle_dir(variant::Symbol)
    get(_PARITY_BUNDLE_DIRS, variant) do
        dir = mktempdir()
        _write_parity_bundle!(dir, variant)
        _PARITY_BUNDLE_DIRS[variant] = dir
        return dir
    end
end

function resolve_parity_bundle_dir(path::AbstractString)
    if path == "parity:posterior"
        return parity_bundle_dir(:posterior)
    elseif path == "parity:full_intrinsic"
        return parity_bundle_dir(:full_intrinsic)
    elseif path == "parity:importance_context"
        return parity_bundle_dir(:importance_context)
    elseif path == "parity:posterior_v2_minimal"
        return parity_bundle_dir(:posterior_v2_minimal)
    elseif path == "parity:w0cdm"
        return parity_bundle_dir(:w0cdm)
    end
    return nothing
end
