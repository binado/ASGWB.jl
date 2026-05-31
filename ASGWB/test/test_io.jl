using HDF5
using Distributions: Uniform, logpdf
using Test
using ASGWB

if !@isdefined parity_bundle_dir
    include(joinpath(@__DIR__, "parity_test_cache.jl"))
end

const _TEST_LOAD_DETS = [Detector("H1"), Detector("L1")]

function _load_variant(variant::Symbol)
    return parity_problem_context(variant, _TEST_LOAD_DETS)
end

@testset "save_bundle/load_bundle round-trip" begin
    grid = FrequencyGrid(1.0, 4.0, 2.0, 1.0, 2.0)
    samples = (
        mass_1_source = [1.4, 1.4],
        mass_2_source = [1.2, 1.2],
        redshift = [0.1, 0.2],
        chi_1 = [0.0, 0.0],
        chi_2 = [0.0, 0.0],
        lambda_1 = [100.0, 100.0],
        lambda_2 = [100.0, 100.0],
        luminosity_distance = [430.0, 880.0]
    )
    fluxes = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    meta = WaveformMetadata("IMRPhenomPV2", :BNS, grid, "deadbeef", "rev", "cmd")
    catalog = WaveformCatalog(samples, fluxes, meta)

    path, io = mktemp()
    close(io)
    try
        save_bundle(path, catalog)
        loaded = load_bundle(path)
        @test Set(keys(loaded.samples)) == Set(keys(catalog.samples))
        for k in keys(catalog.samples)
            @test loaded.samples[k] ≈ catalog.samples[k]
        end
        @test loaded.fluxes ≈ catalog.fluxes
        @test loaded.metadata.approximant == meta.approximant
        @test loaded.metadata.source_type == meta.source_type
        @test loaded.metadata.model_sha256 == meta.model_sha256
        @test loaded.metadata.grid.duration == grid.duration
        @test loaded.metadata.grid.sampling_frequency == grid.sampling_frequency
        @test loaded.metadata.grid.minimum_frequency == grid.minimum_frequency
        @test loaded.metadata.grid.maximum_frequency == grid.maximum_frequency
    finally
        rm(path; force = true)
    end
end

@testset "save_model_toml/load_model_toml round-trip" begin
    cases = (
        (
            ModifiedPropagation{LambdaCDM},
            ParityBNSPopulation(),
            (H0 = 67.4, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5),
            "ModifiedPropagation{LambdaCDM}"
        ),
        (
            ModifiedPropagation{W0CDM},
            ParityBNSPopulation(),
            (H0 = 67.4, Ωm = 0.315, w0 = -0.9, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0,
                zpeak = 2.5),
            "ModifiedPropagation{W0CDM}"
        ),
        (
            ModifiedPropagation{W0WaCDM},
            ParityBNSPopulation(),
            (H0 = 67.4, Ωm = 0.315, w0 = -0.9, wa = 0.1, Ξ₀ = 1.0, Ξₙ = 0.0,
                γ = 2.7, κ = 3.0, zpeak = 2.5),
            "ModifiedPropagation{W0WaCDM}"
        )
    )

    for (C, pop, Λ_raw, tag) in cases
        order = full_hyperparameters(C, pop)
        Λ = canonical_hyperparameters(order, Λ_raw)
        path = joinpath(mktempdir(), "model.toml")
        save_model_toml(path, C, pop, Λ, PARITY_REGISTRY)
        C2, pop2, Λ2 = load_model_toml(path, PARITY_REGISTRY)
        @testset "$tag" begin
            @test C2 === C
            @test pop2 isa ParityBNSPopulation
            @test Λ2 == Λ
        end
    end
end

@testset "load_model_toml requires [parameters]" begin
    path = joinpath(mktempdir(), "model.toml")
    write(path,
        """
        [model]
        cosmology = "ModifiedPropagation{LambdaCDM}"
        population = "bns"
        """)
    @test_throws ArgumentError load_model_toml(path, PARITY_REGISTRY)
end

@testset "read_population requires a registered population" begin
    path = joinpath(mktempdir(), "model.toml")
    write(path,
        """
        [model]
        cosmology = "ModifiedPropagation{LambdaCDM}"
        population = "unregistered"

        [parameters]
        H0 = 67.0
        """)
    @test_throws ArgumentError load_model_toml(path, PARITY_REGISTRY)
end

@testset "ImportanceSamplingProblem is a pure spec" begin
    loaded = _load_variant(:importance_context)
    problem = loaded.problem

    @test problem isa ImportanceSamplingProblem
    @test problem.population_model isa ParityBNSPopulation
    @test redshift(problem) ≈ [0.1, 0.2]
    @test Vector(problem.samples.mass[1, :]) ≈ [1.4, 1.4]
    @test Vector(problem.samples.mass[2, :]) ≈ [1.2, 1.2]
    @test problem.samples.χ₁ ≈ [0.0, 0.0]
    @test problem.samples.Λ₁ ≈ [100.0, 100.0]
    @test problem.fluxes ≈ Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]

    Λ = fiducial_hyperparameters(problem)
    @test Λ.H0 == 67.0
    @test Λ.Ωm == 0.315
    @test Λ.Ξ₀ == 1.0
    @test Λ.γ == 2.7
end

@testset "build_model_context reconstructs derived caches" begin
    loaded = _load_variant(:importance_context)
    problem = loaded.problem
    C = loaded.cosmology_type
    ctx = loaded.ctx
    pop = problem.population_model
    Λ = fiducial_hyperparameters(problem)
    z = problem.samples.redshift

    @test ctx.dgw_fid_sq ≈ ASGWB._reconstruct_dgw_fid_sq(z, C, Λ)
    @test ctx.proposal_log_prob ≈
          ASGWB._reconstruct_proposal_log_prob(problem.samples, C, pop, Λ)
    # Ξ₀ = 1, Ξₙ = 0 ⇒ D_gw = D_L ⇒ no rescaling of the raw fluxes.
    @test ctx.cached_flux_over_dgw2 ≈ problem.fluxes
    @test ctx.cached_flux_over_dgw2 ≈
          ASGWB._reconstruct_cached_flux_over_dgw2(problem.fluxes, z, C, Λ)

    @test length(ctx.redshift_grid) == length(DEFAULT_Z_GRID)
    @test ctx.observation.frequencies ≈ [0.0, 20.0, 40.0]
    @test ctx.observation.in_band_mask == BitVector([false, true, true])
    @test length(ctx.observation.effective_psd) == length(ctx.observation.frequencies)
    @test ctx.observation.sgwb_scale_in_band ≈
          ctx.observation.sgwb_scale[ctx.observation.in_band_mask]
    @test ctx.observation.observation_time_yr == 1.0
    @test ctx.observation.observation_time_sec ≈ 365.25 * 24 * 3600
    @test ctx.local_merger_rate == 161.0

    @test all(isfinite, ctx.fiducial_spectral_density)
    @test length(ctx.fiducial_spectral_density) == length(ctx.observation.frequencies)
    @test fiducial_redshift_integral(problem, C) ≈
          fiducial_redshift_integral(C, pop, Λ) rtol = 1e-6
end

@testset "fiducial spectral density differs across cosmologies" begin
    loaded_w0 = _load_variant(:w0cdm)
    @test loaded_w0.cosmology_type === ModifiedPropagation{W0CDM}
    fs_w0 = loaded_w0.ctx.fiducial_spectral_density
    @test all(isfinite, fs_w0)

    # Build a LambdaCDM context from the same raw inputs and confirm the fiducial spectrum
    # is not identical (the cosmology genuinely feeds through the caches).
    p_w0 = loaded_w0.problem
    Λ = fiducial_hyperparameters(p_w0)
    C_lcdm = ModifiedPropagation{LambdaCDM}
    order_lcdm = full_hyperparameters(C_lcdm, p_w0.population_model)
    Λ_lcdm = canonical_hyperparameters(order_lcdm, (; (k => Λ[k] for k in order_lcdm)...))
    p_lcdm = ImportanceSamplingProblem(
        p_w0.population_model, p_w0.fluxes, p_w0.samples, Λ_lcdm)
    grid = FrequencyGrid(0.05, 80.0, 20.0, 15.0, 45.0)
    ctx_lcdm = build_model_context(
        p_lcdm, C_lcdm, grid, _TEST_LOAD_DETS, 1.0, 161.0)
    @test !(fs_w0 ≈ ctx_lcdm.fiducial_spectral_density)
end

@testset "verify_model_fingerprint throws on mismatch" begin
    dir = parity_bundle_dir(:importance_context)
    catalog = load_bundle(joinpath(dir, "bundle.h5"))
    tmp_dir = mktempdir()
    bad_toml = joinpath(tmp_dir, "model.toml")
    write(bad_toml,
        """
        [model]
        cosmology = "ModifiedPropagation{LambdaCDM}"

        [parameters]
        H0 = 99.0
        "Ωm" = 0.5
        "Ξ₀" = 1.0
        "Ξₙ" = 0.0
        "γ" = 2.7
        "κ" = 3.0
        zpeak = 2.5
        """)
    @test_throws ArgumentError verify_model_fingerprint(catalog, bad_toml)
end
