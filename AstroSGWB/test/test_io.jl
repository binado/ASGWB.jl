using HDF5
using NPZ
using Distributions: Uniform, logpdf
using Test
using AstroSGWB
import PlusCross

if !@isdefined parity_catalog_dir
    include(joinpath(@__DIR__, "parity_test_cache.jl"))
end

const _TEST_LOAD_DETS = [Detector("H1"), Detector("L1")]

function _load_variant(variant::Symbol)
    return parity_problem_context(variant, _TEST_LOAD_DETS)
end

@testset "waveform_catalog v1 round-trip" begin
    samples = (
        mass_1_source = [1.4, 1.4],
        mass_2_source = [1.2, 1.2],
        redshift = [0.1, 0.2],
        chi_1 = [0.0, 0.0],
        chi_2 = [0.0, 0.0],
        lambda_1 = [100.0, 100.0],
        lambda_2 = [100.0, 100.0],
        luminosity_distance = [430.0, 880.0],
        inclination = [0.0, 0.0]
    )
    # Both polarizations carry power here, unlike the parity fixtures, so the
    # `|h₊|² + |h×|²` reduction is exercised rather than trivially `|h₊|²`.
    plus = ComplexF64[0.0 0.0; 1.0+0.0im 0.0+1.0im; 2.0+0.0im 1.0+1.0im]
    cross = ComplexF64[0.0 0.0; 0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+2.0im]
    expected_fluxes = abs2.(plus) .+ abs2.(cross)

    path, io = mktemp()
    close(io)
    try
        PlusCross.save_catalog(
            path,
            PlusCross.WaveformCatalog(;
                frequencies = [0.0, 1.0, 2.0],
                plus = plus,
                cross = cross,
                source_parameters = samples,
                approximant = "IMRPhenomPV2",
                minimum_frequency = 1.0,
                maximum_frequency = 2.0,
                reference_frequency = 2.0,
                sampling_frequency = 4.0
            )
        )
        catalog = load_catalog(path)

        @test catalog isa SGWBCatalog
        @test Set(keys(catalog.samples)) == Set(keys(samples))
        for k in keys(samples)
            @test catalog.samples[k] == samples[k]
        end
        @test catalog.fluxes == expected_fluxes
        @test catalog.frequencies == [0.0, 1.0, 2.0]
        @test catalog.approximant == "IMRPhenomPV2"
        @test size(catalog.fluxes) == (3, 2)

        # The polarizations themselves survive HDF5 byte-for-byte; only the
        # derived flux is subject to the reduction's rounding.
        raw = PlusCross.load_catalog(path)
        @test raw.plus == plus
        @test raw.cross == cross
    finally
        rm(path; force = true)
    end
end

@testset "load_catalog rejects a non-v1 file" begin
    path, io = mktemp()
    close(io)
    try
        HDF5.h5open(path, "w") do f
            HDF5.attributes(f)["format_name"] = "something_else"
        end
        @test_throws ArgumentError load_catalog(path)
    finally
        rm(path; force = true)
    end
end

# Cross-language parity: both repos read the *same* file. Regenerate from the
# repo root with the astrogwb checkout's interpreter (see the script docstring)::
#   ../astrogwb/.venv/bin/python3 scripts/generate_catalog_parity_fixture.py
@testset "flux reduction matches the Python astrogwb stack" begin
    h5_path = joinpath(@__DIR__, "fixtures", "catalog_parity_reference.h5")
    npz_path = joinpath(@__DIR__, "fixtures", "catalog_parity_reference.npz")
    if !(isfile(h5_path) && isfile(npz_path))
        # Fixtures are not committed (see comment above).
        @test_skip false
    else
        catalog = load_catalog(h5_path)
        reference = NPZ.npzread(npz_path)

        # `polarization_power` already returns `(nfreq, nsamples)`, the same
        # orientation HDF5.jl gives Julia, so no transpose is involved.
        @test size(catalog.fluxes) == size(reference["fluxes"])
        @test catalog.frequencies ≈ vec(reference["frequencies"])

        # Julia's `abs2(z)` computes `re² + im²`; NumPy's `abs(z)**2` squares a
        # `hypot`, so the two agree to a few ulp rather than bit-for-bit.
        @test catalog.fluxes≈reference["fluxes"] rtol=1.0e-13
    end
end

@testset "average_mode is derived from the inclination column" begin
    @test average_mode(load_catalog(joinpath(
        parity_catalog_dir(:importance_context), "catalog.h5"))) ===
          AnalyticInclination()
    @test average_mode(load_catalog(joinpath(
        parity_catalog_dir(:sampled_inclination), "catalog.h5"))) ===
          CatalogInclination()

    # A catalog with no `inclination` column falls back to the face-on
    # convention of the legacy generator.
    no_column = SGWBCatalog(
        [0.0, 1.0], zeros(2, 2), (redshift = [0.1, 0.2],),
        "IMRPhenomPV2")
    @test average_mode(no_column) === AnalyticInclination()
end

@testset "catalog inputs are explicit" begin
    loaded = _load_variant(:importance_context)

    @test redshift(loaded.samples) ≈ [0.1, 0.2]
    @test loaded.samples.luminosity_distance ≈ [430.0, 880.0]
    @test loaded.fluxes ≈ Float64[1.0 1.5; 2.0 2.5]

    Λ = loaded.fiducials
    @test Λ.H0 == 67.0
    @test Λ.Ωm == 0.315
    @test Λ.Ξ₀ == 1.0
    @test Λ.γ == 2.7
end

@testset "parity context constructs catalog and observation data" begin
    loaded = _load_variant(:importance_context)
    observation = loaded.observation

    @test all(isfinite, loaded.samples.luminosity_distance)
    @test all(>(0), loaded.samples.luminosity_distance)

    @test observation.frequencies ≈ [20.0, 40.0]
    @test length(observation.effective_psd) == length(observation.frequencies)
    @test length(observation.sgwb_scale) == length(observation.frequencies)
    @test observation.observation_time == 1.0
    @test year_to_second(observation.observation_time) ≈ 365.25 * 24 * 3600
end
