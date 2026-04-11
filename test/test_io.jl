using Test

@testset "importance_sampling_problem matches load_cache fixture" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "importance_context_julia.h5")
    from_file = load_cache(fixture_path)

    sgwb_scale = [1 / sqrt(63115200.0), 1 / sqrt(63115200.0)]
    proposal = ProposalData(
        ["redshift"],
        Dict("redshift" => [0.1, 0.2]),
        [0.0, 0.0],
        reshape([0.1, 0.2], :, 1),
        [1.0 2.0; 1.5 2.5],
        [4.0, 9.0],
    )
    observation = ObservationConfig(
        [1.0, 2.0],
        [1.0, 1.0],
        sgwb_scale,
        BitVector([true, true]),
        [0.0, 0.0],
        365.25 * 24 * 3600,
        1.0,
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 1024, nothing)
    hyper = HyperParameters(; H0=67.0, Omega_m=0.315, chi0=1.0, chin=0.0)
    from_memory = importance_sampling_problem(
        proposal,
        observation,
        spec,
        161.0,
        1.0,
        hyper,
    )

    @test from_memory.proposal.intrinsic_site_order == from_file.proposal.intrinsic_site_order
    @test from_memory.proposal.samples == from_file.proposal.samples
    @test from_memory.proposal.log_prob ≈ from_file.proposal.log_prob
    @test from_memory.proposal.intrinsic_vector ≈ from_file.proposal.intrinsic_vector
    @test from_memory.proposal.cached_flux_over_dgw2 ≈ from_file.proposal.cached_flux_over_dgw2
    @test from_memory.proposal.dgw_fid_sq ≈ from_file.proposal.dgw_fid_sq
    @test from_memory.observation.frequencies ≈ from_file.observation.frequencies
    @test from_memory.observation.covariance ≈ from_file.observation.covariance
    @test from_memory.observation.sgwb_scale ≈ from_file.observation.sgwb_scale
    @test from_memory.observation.in_band_mask == from_file.observation.in_band_mask
    @test from_memory.observation.fiducial_spectral_density ≈
        from_file.observation.fiducial_spectral_density
    @test from_memory.observation.observation_time_sec == from_file.observation.observation_time_sec
    @test from_memory.observation.observation_time_yr == from_file.observation.observation_time_yr
    @test from_memory.redshift_prior_spec.family == from_file.redshift_prior_spec.family
    @test from_memory.redshift_prior_spec.z_min == from_file.redshift_prior_spec.z_min
    @test from_memory.redshift_prior_spec.z_max == from_file.redshift_prior_spec.z_max
    @test from_memory.redshift_prior_spec.num_interp == from_file.redshift_prior_spec.num_interp
    @test from_memory.redshift_prior_spec.time_delay_model ===
        from_file.redshift_prior_spec.time_delay_model
    @test from_memory.hyperparameters.H0 == from_file.hyperparameters.H0
    @test from_memory.hyperparameters.Omega_m == from_file.hyperparameters.Omega_m
    @test from_memory.hyperparameters.chi0 == from_file.hyperparameters.chi0
    @test from_memory.hyperparameters.chin == from_file.hyperparameters.chin
    @test from_memory.local_merger_rate == from_file.local_merger_rate
    @test from_memory.redshift_integral_fiducial == from_file.redshift_integral_fiducial
    @test typeof(from_memory.strategy) == typeof(from_file.strategy)
end

@testset "load_cache" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "importance_context_julia.h5")
    problem = load_cache(fixture_path)

    @test problem.proposal.intrinsic_site_order == ["redshift"]
    @test problem.proposal.samples["redshift"] ≈ [0.1, 0.2]
    @test problem.proposal.log_prob ≈ [0.0, 0.0]
    @test problem.proposal.intrinsic_vector ≈ reshape([0.1, 0.2], :, 1)
    @test problem.proposal.cached_flux_over_dgw2 ≈ [1.0 2.0; 1.5 2.5]
    @test problem.proposal.dgw_fid_sq ≈ [4.0, 9.0]
    @test problem.observation.frequencies ≈ [1.0, 2.0]
    @test problem.observation.covariance ≈ [1.0, 1.0]
    @test problem.observation.sgwb_scale ≈ [1 / sqrt(63115200.0), 1 / sqrt(63115200.0)]
    @test problem.observation.in_band_mask == BitVector([true, true])
    @test problem.observation.fiducial_spectral_density ≈ [0.0, 0.0]
    @test problem.observation.sgwb_scale_in_band ≈ problem.observation.sgwb_scale
    @test problem.observation.fiducial_spectral_density_in_band ≈ [0.0, 0.0]
    @test problem.hyperparameters.H0 == 67.0
    @test problem.hyperparameters.Omega_m == 0.315
    @test problem.hyperparameters.chi0 == 1.0
    @test problem.hyperparameters.chin == 0.0
    @test problem.redshift_prior_spec.family == MadauDickinson
    @test problem.redshift_prior_spec.z_min == 0.001
    @test problem.redshift_prior_spec.z_max == 20.0
    @test problem.redshift_prior_spec.time_delay_model === nothing
    @test problem.redshift_prior_spec.num_interp == 1024
    @test problem.local_merger_rate == 161.0
    @test problem.observation.observation_time_yr == 1.0
    @test problem.observation.observation_time_sec == 365.25 * 24 * 3600
    @test problem.redshift_integral_fiducial == 1.0
    @test problem.strategy isa RedshiftOnly
    @test redshift(problem) ≈ [0.1, 0.2]
end
