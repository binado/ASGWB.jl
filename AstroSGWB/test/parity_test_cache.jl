# Test-only synthetic catalog fixtures. Include after `using AstroSGWB` (see `runtests.jl`).

import PlusCross

if !@isdefined ParityBNSPopulation
    include(joinpath(@__DIR__, "fixture_population.jl"))
end

const _PARITY_APPROXIMANT = "IMRPhenomPV2_NRTidalv2"

# The frequency axis and band edges are stored in the catalog file, not derived
# from `(duration, sampling_frequency)`, so they are written out literally here.
# The DC bin (f = 0) is out of band; callers slice it off (see
# `parity_problem_context`).
const _PARITY_FREQUENCIES = [0.0, 20.0, 40.0]
const _PARITY_MINIMUM_FREQUENCY = 15.0
const _PARITY_MAXIMUM_FREQUENCY = 40.0
const _PARITY_REFERENCE_FREQUENCY = 20.0
const _PARITY_SAMPLING_FREQUENCY = 80.0

function _parity_hyperparameters(C, P, pop, overrides::NamedTuple = NamedTuple())
    defaults = (H0 = 67.0, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5)
    order = full_hyperparameters(C, P, pop)
    return canonical_hyperparameters(order, merge(defaults, overrides))
end

function _parity_hyperparameters_w0(C, P, pop, overrides::NamedTuple = NamedTuple())
    defaults = (H0 = 67.0, Ωm = 0.315, w0 = -0.9, Ξ₀ = 1.0, Ξₙ = 0.0,
        γ = 2.7, κ = 3.0, zpeak = 2.5)
    order = full_hyperparameters(C, P, pop)
    return canonical_hyperparameters(order, merge(defaults, overrides))
end

function _write_parity_catalog!(dir::String, variant::Symbol)
    if variant == :posterior
        _write_posterior_catalog(dir)
    elseif variant == :full_intrinsic
        _write_full_intrinsic_catalog(dir)
    elseif variant == :importance_context || variant == :posterior_v2_minimal
        _write_importance_context_catalog(dir)
    elseif variant == :sampled_inclination
        _write_sampled_inclination_catalog(dir)
    elseif variant == :w0cdm
        _write_w0cdm_catalog(dir)
    else
        throw(ArgumentError("unknown parity catalog variant $(repr(variant))"))
    end
    return dir
end

"""
    parity_polarizations(cached_flux) -> (plus, cross)

Synthesize complex polarizations whose power `|h₊|² + |h×|²` reproduces
`cached_flux`. The v1 format stores the fundamental artifact rather than the
reduction, so fixtures that want a particular flux must back-solve for one.

Putting all the power in `h₊` makes `plus = sqrt(cached_flux)` the obvious
choice; note that `abs2 ∘ sqrt` is only bit-exact when the square root is
exactly representable (0.0, 1.0, 3.5, 4.0 among the values used here) and is
otherwise correct to 1 ulp, so compare recovered fluxes with `≈`.
"""
function parity_polarizations(cached_flux::AbstractMatrix{<:Real})
    return ComplexF64.(sqrt.(cached_flux)), zeros(ComplexF64, size(cached_flux))
end

"""
    _write_catalog_h5(dir, samples, cached_flux; inclination=nothing) -> String

Write a `waveform_catalog` v1 fixture reducing to `cached_flux`. `inclination`
defaults to an all-zero column, so [`average_mode`](@ref) derives
`AnalyticInclination()`; pass a non-zero column to exercise the other branch.
"""
function _write_catalog_h5(dir, samples::NamedTuple, cached_flux::AbstractMatrix{<:Real};
        inclination = nothing)
    path = joinpath(dir, "catalog.h5")
    n = size(cached_flux, 2)
    incl = isnothing(inclination) ? zeros(n) : collect(Float64, inclination)
    plus, cross = parity_polarizations(cached_flux)
    catalog = PlusCross.WaveformCatalog(;
        frequencies = _PARITY_FREQUENCIES,
        plus = plus,
        cross = cross,
        source_parameters = merge(samples, (inclination = incl,)),
        approximant = _PARITY_APPROXIMANT,
        minimum_frequency = _PARITY_MINIMUM_FREQUENCY,
        maximum_frequency = _PARITY_MAXIMUM_FREQUENCY,
        reference_frequency = _PARITY_REFERENCE_FREQUENCY,
        sampling_frequency = _PARITY_SAMPLING_FREQUENCY
    )
    PlusCross.save_catalog(path, catalog)
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

function _write_posterior_catalog(dir)
    C, P = LambdaCDM, ModifiedPropagation
    pop = ParityBNSPopulation()
    Λ = _parity_hyperparameters(C, P, pop, (γ = 2.7, κ = 3.0, zpeak = 2.0))

    samples = _make_bns_samples(
        [1.4, 1.4], [1.2, 1.2], [0.1, 0.2];
        luminosity_distances = [430.0, 880.0]
    )
    cached_flux = Float64[0.0 0.0; 1.0 4.0; 2.0 5.0]
    _write_catalog_h5(dir, samples, cached_flux)
    return dir
end

function _write_full_intrinsic_catalog(dir)
    C, P = LambdaCDM, ModifiedPropagation
    pop = ParityBNSPopulation()
    Λ = _parity_hyperparameters(C, P, pop, (γ = 2.7, κ = 3.0, zpeak = 2.0))

    samples = _make_bns_samples(
        [1.8, 2.2, 1.4, 2.4], [1.2, 1.7, 1.1, 1.3], [0.1, 0.2, 0.3, 0.5];
        chi1 = [0.0, 0.2, -0.1, 0.5],
        chi2 = [0.1, -0.2, 0.0, 0.3],
        lambda1 = [400.0, 800.0, 1200.0, 2000.0],
        lambda2 = [300.0, 600.0, 700.0, 1500.0],
        luminosity_distances = [430.0, 880.0, 1350.0, 2300.0]
    )
    cached_flux = Float64[0.0 0.0 0.0 0.0
                          1.0 1.5 2.0 2.5
                          2.0 2.5 3.0 3.5]
    _write_catalog_h5(dir, samples, cached_flux)
    return dir
end

function _write_importance_context_catalog(dir)
    C, P = LambdaCDM, ModifiedPropagation
    pop = ParityBNSPopulation()
    Λ = _parity_hyperparameters(C, P, pop, (γ = 2.7, κ = 3.0, zpeak = 2.5))

    samples = _make_bns_samples(
        [1.4, 1.4], [1.2, 1.2], [0.1, 0.2];
        luminosity_distances = [430.0, 880.0]
    )
    cached_flux = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    _write_catalog_h5(dir, samples, cached_flux)
    return dir
end

"""
Same payload as [`_write_importance_context_catalog`](@ref) but with a sampled
`inclination` column, so [`average_mode`](@ref) derives `CatalogInclination()`.
"""
function _write_sampled_inclination_catalog(dir)
    samples = _make_bns_samples(
        [1.4, 1.4], [1.2, 1.2], [0.1, 0.2];
        luminosity_distances = [430.0, 880.0]
    )
    cached_flux = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    _write_catalog_h5(dir, samples, cached_flux; inclination = [0.0, 0.7])
    return dir
end

function _write_w0cdm_catalog(dir)
    C, P = W0CDM, ModifiedPropagation
    pop = ParityBNSPopulation()
    Λ = _parity_hyperparameters_w0(C, P, pop, (γ = 2.7, κ = 3.0, zpeak = 2.5))

    samples = _make_bns_samples(
        [1.4, 1.4], [1.2, 1.2], [0.1, 0.2];
        luminosity_distances = [430.0, 880.0]
    )
    cached_flux = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    _write_catalog_h5(dir, samples, cached_flux)
    return dir
end

const _PARITY_CATALOG_DIRS = Dict{Symbol, String}()

# The local merger rate used to travel with this: S7 made it the live `Λ.R₀`, which
# belongs to the importance model's hyperparameters, not to detector state.
function parity_observation_kwargs(variant::Symbol)
    if variant == :posterior || variant == :full_intrinsic
        return (observation_time = 1e-6,)
    else
        return (observation_time = 1.0,)
    end
end

"""
    parity_bns_samples_from_catalog(catalog_samples) -> NamedTuple

Test-side mirror of the caller's slim BNS sample restructuring: keep only the `redshift`
and `luminosity_distance` columns the importance-weight loop reads.
"""
function parity_bns_samples_from_catalog(catalog_samples::NamedTuple)
    return (
        redshift = copy(catalog_samples.redshift),
        luminosity_distance = copy(catalog_samples.luminosity_distance)
    )
end

"""
    parity_problem_context(variant, detectors)
        -> (; fluxes, samples, fiducials, observation)

Load the parity catalog for `variant`, restructure its samples, and build its
[`ObservationContext`](@ref). Physical importance-model preparation is tested by
`AstroSGWBImportanceModels`; this core fixture only owns catalog, sample, and observation
data.
"""
function parity_problem_context(variant::Symbol, detectors)
    dir = parity_catalog_dir(variant)
    catalog = load_catalog(joinpath(dir, "catalog.h5"))
    pop = ParityBNSPopulation()
    C = variant == :w0cdm ? W0CDM : LambdaCDM
    P = ModifiedPropagation
    Λ = variant == :w0cdm ?
        _parity_hyperparameters_w0(C, P, pop, (γ = 2.7, κ = 3.0, zpeak = 2.5)) :
        if variant == :posterior || variant == :full_intrinsic
        _parity_hyperparameters(C, P, pop, (γ = 2.7, κ = 3.0, zpeak = 2.0))
    else
        _parity_hyperparameters(C, P, pop, (γ = 2.7, κ = 3.0, zpeak = 2.5))
    end
    samples = parity_bns_samples_from_catalog(catalog.samples)
    # Re-reference the stored EM-distance fluxes to the fiducial GW distance, matching the
    # `+2 log Ξ_fid` term the importance model's log-weights carry. Every parity variant
    # uses Ξ₀ = 1, so this is currently a no-op; the bang form is safe because each call
    # re-reads `catalog.h5` from scratch.
    apply_gw_distance_correction!(catalog, propagation(P, Λ))
    kw = parity_observation_kwargs(variant)
    # Band selection is the caller's job: keep every bin above DC, matching the
    # band edges stored in the fixture files.
    band = catalog.frequencies .> 0.0
    fluxes = catalog.fluxes[band, :]
    observation = build_observation_context(
        catalog.frequencies[band], Vector{Detector}(collect(detectors)),
        kw.observation_time)
    return (;
        fluxes = fluxes,
        samples = samples,
        fiducials = Λ,
        observation = observation,
        average_mode = average_mode(catalog))
end

"""
    parity_catalog_dir(variant) -> String

Return the directory containing `catalog.h5` for `variant`.
The catalog is generated lazily on first call.

Variants: `:posterior`, `:full_intrinsic`, `:importance_context`,
`:posterior_v2_minimal` (alias for `:importance_context`), `:sampled_inclination`,
`:w0cdm`.
"""
function parity_catalog_dir(variant::Symbol)
    get(_PARITY_CATALOG_DIRS, variant) do
        dir = mktempdir()
        _write_parity_catalog!(dir, variant)
        _PARITY_CATALOG_DIRS[variant] = dir
        return dir
    end
end

function resolve_parity_catalog_dir(path::AbstractString)
    if path == "parity:posterior"
        return parity_catalog_dir(:posterior)
    elseif path == "parity:full_intrinsic"
        return parity_catalog_dir(:full_intrinsic)
    elseif path == "parity:importance_context"
        return parity_catalog_dir(:importance_context)
    elseif path == "parity:posterior_v2_minimal"
        return parity_catalog_dir(:posterior_v2_minimal)
    elseif path == "parity:sampled_inclination"
        return parity_catalog_dir(:sampled_inclination)
    elseif path == "parity:w0cdm"
        return parity_catalog_dir(:w0cdm)
    end
    return nothing
end
