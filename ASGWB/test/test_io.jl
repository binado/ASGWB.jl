using HDF5
using Distributions: Uniform, logpdf
using Test
using ASGWB

if !@isdefined parity_bundle_dir
    include(joinpath(@__DIR__, "parity_test_cache.jl"))
end

const _TEST_LOAD_DETS = [Detector("H1"), Detector("L1")]

function _load_variant(variant::Symbol)
    dir = parity_bundle_dir(variant)
    return load_problem(
        joinpath(dir, "bundle.h5"),
        joinpath(dir, "model.toml"),
        _TEST_LOAD_DETS;
        parity_observation_kwargs(variant)...
    )
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

@testset "save_model_config/load_model_config round-trip" begin
    cases = (
        (
            MadauDickinsonModifiedPropagation(),
            (H0 = 67.4, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5),
            "LambdaCDM"
        ),
        (
            MadauDickinsonModifiedPropagation{W0CDM}(),
            (H0 = 67.4, Ωm = 0.315, w0 = -0.9, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0,
                zpeak = 2.5),
            "W0CDM"
        ),
        (
            MadauDickinsonModifiedPropagation{W0WaCDM}(),
            (H0 = 67.4, Ωm = 0.315, w0 = -0.9, wa = 0.1, Ξ₀ = 1.0, Ξₙ = 0.0,
                γ = 2.7, κ = 3.0, zpeak = 2.5),
            "W0WaCDM"
        )
    )

    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    for (model, Λ, tag) in cases
        config = ModelConfig(model, canonical_hyperparameters(model, Λ), spec)
        path = joinpath(mktempdir(), "model.toml")
        save_model_config(path, config)
        loaded = load_model_config(path)
        @testset "$tag" begin
            @test typeof(loaded.model) == typeof(model)
            @test loaded.fiducial_hyperparameters == config.fiducial_hyperparameters
            @test loaded.redshift_prior_spec.family == spec.family
            @test loaded.redshift_prior_spec.z_min == spec.z_min
            @test loaded.redshift_prior_spec.z_max == spec.z_max
            @test loaded.redshift_prior_spec.num_interp == spec.num_interp
            @test loaded.redshift_prior_spec.time_delay_model === nothing
        end
    end
end

@testset "load_model_config requires [parameters]" begin
    path = joinpath(mktempdir(), "model.toml")
    write(path,
        """
        [model]
        name = "madau_dickinson_modified_propagation"
        cosmology = "LambdaCDM"

        [redshift]
        z_min = 0.001
        z_max = 20.0
        num_interp = 64
        time_delay_model = "none"
        """)
    @test_throws ArgumentError load_model_config(path)
end

@testset "load_problem reconstructs derived fields" begin
    problem = _load_variant(:importance_context)
    z = problem.proposal.samples.redshift
    Λ = fiducial_hyperparameters(problem)
    spec = problem.redshift_prior_spec

    expected_dgw_sq = reconstruct_dgw_fid_sq(z, problem.model, Λ)
    expected_lp = reconstruct_proposal_log_prob(problem.proposal.samples, spec, problem.model, Λ)
    expected_ri = fiducial_redshift_integral(problem.model, Λ, spec)

    @test problem.proposal.dgw_fid_sq ≈ expected_dgw_sq
    @test problem.proposal.log_prob ≈ expected_lp
    @test fiducial_redshift_integral(problem) ≈ expected_ri rtol = 1e-6
    @test fiducial_spectral_density(problem) ≈ problem.observation.fiducial_spectral_density
end

@testset "importance_sampling_problem matches load_problem fixture" begin
    from_file = _load_variant(:importance_context)
    model = from_file.model
    Λ = from_file.fiducial_hyperparameters
    spec = from_file.redshift_prior_spec
    samples = (
        mass = stack_source_masses([1.4, 1.4], [1.2, 1.2]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.0],
        χ₂ = [0.0, 0.0],
        Λ₁ = [100.0, 100.0],
        Λ₂ = [100.0, 100.0]
    )
    lp = reconstruct_proposal_log_prob(samples, spec, model, Λ)
    intrinsic_mat = Float64[1.4 1.2 0.1 0.0 0.0 100.0 100.0
                            1.4 1.2 0.2 0.0 0.0 100.0 100.0]
    dgw_sq = reconstruct_dgw_fid_sq(samples.redshift, model, Λ)
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        lp,
        intrinsic_mat,
        from_file.proposal.cached_flux_over_dgw2,
        dgw_sq
    )
    from_memory = importance_sampling_problem(
        proposal,
        from_file.observation,
        model,
        Λ,
        spec,
        from_file.local_merger_rate
    )

    @test from_memory.proposal.intrinsic_site_order ==
          from_file.proposal.intrinsic_site_order
    @test from_memory.proposal.samples.redshift == from_file.proposal.samples.redshift
    @test from_memory.proposal.log_prob ≈ from_file.proposal.log_prob
    @test from_memory.proposal.intrinsic_vector ≈ from_file.proposal.intrinsic_vector
    @test from_memory.proposal.cached_flux_over_dgw2 ≈
          from_file.proposal.cached_flux_over_dgw2
    @test from_memory.proposal.dgw_fid_sq ≈ from_file.proposal.dgw_fid_sq
    @test from_memory.observation.frequencies ≈ from_file.observation.frequencies
    @test from_memory.observation.effective_psd ≈ from_file.observation.effective_psd
    @test from_memory.observation.sgwb_scale ≈ from_file.observation.sgwb_scale
    @test from_memory.observation.in_band_mask == from_file.observation.in_band_mask
    @test from_memory.observation.fiducial_spectral_density ≈
          from_file.observation.fiducial_spectral_density
    @test from_memory.redshift_prior_spec.num_interp ==
          from_file.redshift_prior_spec.num_interp
    @test from_memory.fiducial_hyperparameters == from_file.fiducial_hyperparameters
    @test from_memory.local_merger_rate == from_file.local_merger_rate
    @test typeof(from_memory.strategy) == typeof(from_file.strategy)
end

@testset "load_problem" begin
    problem = _load_variant(:importance_context)

    @test problem.proposal.intrinsic_site_order == FULL_BNS_INTRINSIC_ORDER
    s = problem.proposal.samples
    @test s.redshift ≈ [0.1, 0.2]
    @test Vector(s.mass[1, :]) ≈ [1.4, 1.4]
    @test Vector(s.mass[2, :]) ≈ [1.2, 1.2]
    @test s.χ₁ ≈ [0.0, 0.0]
    @test s.χ₂ ≈ [0.0, 0.0]
    @test s.Λ₁ ≈ [100.0, 100.0]
    @test s.Λ₂ ≈ [100.0, 100.0]

    expected_lp = reconstruct_proposal_log_prob(
        problem.proposal.samples,
        problem.redshift_prior_spec,
        problem.model,
        fiducial_hyperparameters(problem)
    )
    @test problem.proposal.log_prob ≈ expected_lp rtol = 1e-6
    @test problem.proposal.intrinsic_vector ≈ Float64[1.4 1.2 0.1 0.0 0.0 100.0 100.0
                  1.4 1.2 0.2 0.0 0.0 100.0 100.0]
    @test problem.proposal.cached_flux_over_dgw2 ≈ Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    @test problem.proposal.dgw_fid_sq ≈ reconstruct_dgw_fid_sq(
        problem.proposal.samples.redshift,
        problem.model,
        fiducial_hyperparameters(problem)
    )

    @test problem.observation.frequencies ≈ [0.0, 20.0, 40.0]
    @test length(problem.observation.effective_psd) ==
          length(problem.observation.frequencies)
    @test length(problem.observation.sgwb_scale) == length(problem.observation.frequencies)
    @test problem.observation.in_band_mask == BitVector([false, true, true])

    ev = evaluate_model_terms(fiducial_hyperparameters(problem), problem)
    @test problem.observation.fiducial_spectral_density ≈ ev.spectral_density
    @test problem.observation.sgwb_scale_in_band ≈
          problem.observation.sgwb_scale[problem.observation.in_band_mask]
    @test problem.observation.fiducial_spectral_density_in_band ≈
          ev.spectral_density_in_band

    Λ = fiducial_hyperparameters(problem)
    @test problem.model isa MadauDickinsonModifiedPropagation{LambdaCDM}
    @test Λ.H0 == 67.0
    @test Λ.Ωm == 0.315
    @test Λ.Ξ₀ == 1.0
    @test Λ.Ξₙ == 0.0
    @test problem.redshift_prior_spec.family == MadauDickinson
    @test problem.redshift_prior_spec.z_min == 0.001
    @test problem.redshift_prior_spec.z_max == 20.0
    @test problem.redshift_prior_spec.time_delay_model === nothing
    @test problem.redshift_prior_spec.num_interp == 1024
    @test problem.local_merger_rate == 161.0
    @test problem.observation.observation_time_yr == 1.0
    @test problem.observation.observation_time_sec ≈ 365.25 * 24 * 3600
    @test fiducial_redshift_integral(problem) ≈
          fiducial_redshift_integral(problem.model, Λ, problem.redshift_prior_spec) rtol = 1e-6
    @test problem.strategy isa FullBNS
    @test redshift(problem) ≈ [0.1, 0.2]
end

@testset "fiducial_spectral_density uses W0CDM bundle" begin
    p = _load_variant(:w0cdm)
    @test p.model isa MadauDickinsonModifiedPropagation{W0CDM}
    Λ = fiducial_hyperparameters(p)
    ev = evaluate_model_terms(Λ, p)
    @test fiducial_spectral_density(p) ≈ ev.spectral_density

    h_lcdm = canonical_hyperparameters(
        MadauDickinsonModifiedPropagation(),
        (; (k => Λ[k] for k in hyperparameters(MadauDickinsonModifiedPropagation()))...)
    )
    ev_lcdm = evaluate_model_terms(MadauDickinsonModifiedPropagation(), h_lcdm, p)
    @test !(fiducial_spectral_density(p) ≈ ev_lcdm.spectral_density)
end

@testset "load_problem throws on model fingerprint mismatch" begin
    dir = parity_bundle_dir(:importance_context)
    bundle_path = joinpath(dir, "bundle.h5")
    tmp_dir = mktempdir()
    bad_toml = joinpath(tmp_dir, "model.toml")
    write(bad_toml,
        """
        [model]
        name = "madau_dickinson_modified_propagation"
        cosmology = "LambdaCDM"

        [parameters]
        H0 = 99.0
        Omega_m = 0.5
        Xi_0 = 1.0
        Xi_n = 0.0
        gamma = 2.7
        kappa = 3.0
        z_peak = 2.5

        [redshift]
        z_min = 0.001
        z_max = 20.0
        num_interp = 64
        time_delay_model = "none"
        """)
    @test_throws ArgumentError load_problem(
        bundle_path,
        bad_toml,
        _TEST_LOAD_DETS;
        parity_observation_kwargs(:importance_context)...
    )
end

@testset "importance_sampling_problem builds redshift cache" begin
    model = MadauDickinsonModifiedPropagation()
    Λ = canonical_hyperparameters(
        model,
        (H0 = 67.0, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5)
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    samples = (
        mass = stack_source_masses([1.4], [1.2]),
        redshift = [0.1],
        χ₁ = [0.0],
        χ₂ = [0.0],
        Λ₁ = [100.0],
        Λ₂ = [100.0]
    )
    lp = reconstruct_proposal_log_prob(samples, spec, model, Λ)
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        lp,
        reshape([1.4, 1.2, 0.1, 0.0, 0.0, 100.0, 100.0], 1, :),
        fill(1.0, 1, 2),
        [1.0]
    )
    observation = ObservationConfig(
        [1.0, 2.0],
        [1.0, 1.0],
        [1.0, 1.0],
        BitVector([true, true]),
        [0.0, 0.0],
        1.0,
        1.0
    )
    p = importance_sampling_problem(proposal, observation, model, Λ, spec, 1.0)
    @test fiducial_redshift_integral(p) ≈ fiducial_redshift_integral(model, Λ, spec) rtol = 1e-10
end

@testset "importance_sampling_problem accepts custom intrinsic prior factory" begin
    model = MadauDickinsonModifiedPropagation()
    Λ = canonical_hyperparameters(
        model,
        (H0 = 67.0, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5)
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 64, nothing)
    samples = (
        mass = stack_source_masses([1.4, 1.5], [1.2, 1.3]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.1],
        χ₂ = [0.0, -0.2],
        Λ₁ = [100.0, 200.0],
        Λ₂ = [150.0, 250.0]
    )
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        zeros(2),
        Float64[1.4 1.2 0.1 0.0 0.0 100.0 150.0
                1.5 1.3 0.2 0.1 -0.2 200.0 250.0],
        ones(2, 2),
        ones(2)
    )
    observation = ObservationConfig(
        [1.0, 2.0],
        [1.0, 1.0],
        [1.0, 1.0],
        BitVector([true, true]),
        [0.0, 0.0],
        1.0,
        1.0
    )
    factory = strategy -> begin
        @test strategy isa FullBNS
        IntrinsicPrior((χ₁ = Uniform(-1.0, 1.0), Λ₁ = Uniform(0.0, 500.0)))
    end

    p = importance_sampling_problem(
        proposal,
        observation,
        model,
        Λ,
        spec,
        1.0;
        intrinsic_prior_factory = factory
    )
    expected = [logpdf(Uniform(-1.0, 1.0), samples.χ₁[i]) +
                logpdf(Uniform(0.0, 500.0), samples.Λ₁[i])
                for i in eachindex(samples.χ₁)]
    @test p.redshift_cache.cached_intrinsic_log_prob ≈ expected
end
