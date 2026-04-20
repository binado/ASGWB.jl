using HDF5
using HDF5: delete_attribute
using Test
using Base.Filesystem: cp

const _TEST_LOAD_DETS = [Detector("H1"), Detector("L1")]

@testset "load_cache omits proposal_log_prob, dgw_fid_sq, fiducial spectrum; full BNS reconstruction" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "importance_context_julia.h5")
    ref = load_cache(fixture_path, _TEST_LOAD_DETS)
    z = ref.proposal.samples.redshift
    spec = ref.redshift_prior_spec
    γ, κ, zp = 2.7, 3.0, 2.5
    fid = ProposalFiducialParameters(;
        H0=ref.fiducial_parameters.H0,
        Omega_m=ref.fiducial_parameters.Omega_m,
        chi0=ref.fiducial_parameters.chi0,
        chin=ref.fiducial_parameters.chin,
        gamma=γ,
        kappa=κ,
        z_peak=zp,
    )
    d_l = luminosity_distance.(z, fid.H0, fid.Omega_m)
    d_gw = gravitational_wave_distance.(z, d_l, fid.chi0, fid.chin)
    scale = (d_l ./ d_gw) .^ 2
    raw_flux = ref.proposal.cached_flux_over_dgw2 ./ reshape(scale, 1, :)
    h = HyperParameters(;
        H0=fid.H0,
        Omega_m=fid.Omega_m,
        chi0=fid.chi0,
        chin=fid.chin,
        gamma=γ,
        kappa=κ,
        z_peak=zp,
    )
    bundle = build_redshift_grid_bundle(h, spec)
    expected_lp = reconstruct_proposal_log_prob(ref.proposal.samples, spec, fid)
    expected_ri = Float64(bundle.norm)

    path, io = mktemp()
    close(io)
    try
        h5open(path, "w") do f
            a = attributes(f)
            a[IMPORTANCE_CACHE_COMMAND_ATTR] = "synthetic test writer"
            a[IMPORTANCE_CACHE_GIT_REVISION_ATTR] = "test"
            a["local_merger_rate"] = ref.local_merger_rate
            a["observation_time_sec"] = ref.observation.observation_time_sec
            a["observation_time_yr"] = ref.observation.observation_time_yr
            write(f, "intrinsic_site_order", ref.proposal.intrinsic_site_order)
            write(
                f,
                "proposal_intrinsic_vector",
                Matrix(permutedims(ref.proposal.intrinsic_vector)),
            )
            write(f, "frequencies", ref.observation.frequencies)
            write(f, "in_band_mask", Vector{Bool}(ref.observation.in_band_mask))
            write(f, "cached_flux", raw_flux)

            g = create_group(f, "proposal_samples")
            attributes(g)[PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR] = PROPOSAL_SAMPLES_SOURCE_TYPE_BNS
            s = ref.proposal.samples
            write(g, "mass_1_source", s.mass_1_source)
            write(g, "mass_2_source", s.mass_2_source)
            write(g, "redshift", s.redshift)
            write(g, "chi_1", s.chi_1)
            write(g, "chi_2", s.chi_2)
            write(g, "lambda_1", s.lambda_1)
            write(g, "lambda_2", s.lambda_2)

            hg = create_group(f, "hyperparameters")
            write(hg, "H0", fid.H0)
            write(hg, "Omega_m", fid.Omega_m)
            write(hg, "chi0", fid.chi0)
            write(hg, "chin", fid.chin)
            write(hg, "gamma", γ)
            write(hg, "kappa", κ)
            write(hg, "z_peak", zp)

            sg = create_group(f, "redshift_prior_spec")
            write(sg, "family", "madau_dickinson")
            write(sg, "z_min", spec.z_min)
            write(sg, "z_max", spec.z_max)
            write(sg, "num_interp", spec.num_interp)
        end

        p = load_cache(path, _TEST_LOAD_DETS)
        @test p.proposal.cached_flux_over_dgw2 ≈ ref.proposal.cached_flux_over_dgw2
        @test p.proposal.dgw_fid_sq ≈ d_gw .^ 2
        @test p.proposal.log_prob ≈ expected_lp
        @test fiducial_spectral_density(p) ≈ p.observation.fiducial_spectral_density
        @test fiducial_redshift_integral(p) ≈ p.redshift_integral_fiducial rtol = 1e-6
        @test p.redshift_integral_fiducial ≈ expected_ri rtol = 1e-6
    finally
        rm(path; force=true)
    end
end

@testset "importance_sampling_problem matches load_cache fixture" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "importance_context_julia.h5")
    from_file = load_cache(fixture_path, _TEST_LOAD_DETS)

    samples = FullBNSSamples(
        [1.4, 1.4],
        [1.2, 1.2],
        [0.1, 0.2],
        [0.0, 0.0],
        [0.0, 0.0],
        [100.0, 100.0],
        [100.0, 100.0],
    )
    lp = reconstruct_proposal_log_prob(
        samples,
        from_file.redshift_prior_spec,
        from_file.fiducial_parameters,
    )
    intrinsic_mat = Float64[
        1.4 1.2 0.1 0.0 0.0 100.0 100.0
        1.4 1.2 0.2 0.0 0.0 100.0 100.0
    ]
    dgw_sq = reconstruct_dgw_fid_sq(samples.redshift, from_file.fiducial_parameters)
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        lp,
        intrinsic_mat,
        [1.0 1.5; 2.0 2.5],
        dgw_sq,
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 1024, nothing)
    from_memory = importance_sampling_problem(
        proposal,
        from_file.observation,
        spec,
        161.0,
        1.0,
        from_file.fiducial_parameters,
    )

    @test from_memory.proposal.intrinsic_site_order == from_file.proposal.intrinsic_site_order
    @test from_memory.proposal.samples.redshift == from_file.proposal.samples.redshift
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
    @test from_memory.fiducial_parameters.H0 == from_file.fiducial_parameters.H0
    @test from_memory.fiducial_parameters.Omega_m == from_file.fiducial_parameters.Omega_m
    @test from_memory.fiducial_parameters.chi0 == from_file.fiducial_parameters.chi0
    @test from_memory.fiducial_parameters.chin == from_file.fiducial_parameters.chin
    @test from_memory.local_merger_rate == from_file.local_merger_rate
    @test from_memory.redshift_integral_fiducial == from_file.redshift_integral_fiducial
    @test typeof(from_memory.strategy) == typeof(from_file.strategy)
end

@testset "load_cache" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "importance_context_julia.h5")
    problem = load_cache(fixture_path, _TEST_LOAD_DETS)

    @test problem.proposal.intrinsic_site_order == FULL_BNS_INTRINSIC_ORDER
    s = problem.proposal.samples
    @test s.redshift ≈ [0.1, 0.2]
    @test s.mass_1_source ≈ [1.4, 1.4]
    @test s.mass_2_source ≈ [1.2, 1.2]
    @test s.chi_1 ≈ [0.0, 0.0]
    @test s.chi_2 ≈ [0.0, 0.0]
    @test s.lambda_1 ≈ [100.0, 100.0]
    @test s.lambda_2 ≈ [100.0, 100.0]
    # tolerance reflects Simpson- vs trapezoid-based bundle norm (see radial_interpolant.jl)
    @test problem.proposal.log_prob ≈ [-17.12928958264864, -15.702803964648165] rtol = 2e-3
    @test problem.proposal.intrinsic_vector ≈ Float64[
        1.4 1.2 0.1 0.0 0.0 100.0 100.0
        1.4 1.2 0.2 0.0 0.0 100.0 100.0
    ]
    @test problem.proposal.cached_flux_over_dgw2 ≈ [1.0 1.5; 2.0 2.5]
    @test problem.proposal.dgw_fid_sq ≈
        reconstruct_dgw_fid_sq(problem.proposal.samples.redshift, problem.fiducial_parameters)
    @test problem.observation.frequencies ≈ [1.0, 2.0]
    @test length(problem.observation.covariance) == length(problem.observation.frequencies)
    @test length(problem.observation.sgwb_scale) == length(problem.observation.frequencies)
    @test problem.observation.in_band_mask == BitVector([true, true])
    @test problem.observation.fiducial_spectral_density ≈ [0.0, 0.0]
    @test problem.observation.sgwb_scale_in_band ≈ problem.observation.sgwb_scale
    @test problem.observation.fiducial_spectral_density_in_band ≈ [0.0, 0.0]
    @test problem.fiducial_parameters.H0 == 67.0
    @test problem.fiducial_parameters.Omega_m == 0.315
    @test problem.fiducial_parameters.chi0 == 1.0
    @test problem.fiducial_parameters.chin == 0.0
    @test problem.redshift_prior_spec.family == MadauDickinson
    @test problem.redshift_prior_spec.z_min == 0.001
    @test problem.redshift_prior_spec.z_max == 20.0
    @test problem.redshift_prior_spec.time_delay_model === nothing
    @test problem.redshift_prior_spec.num_interp == 1024
    @test problem.local_merger_rate == 161.0
    @test problem.observation.observation_time_yr == 1.0
    @test problem.observation.observation_time_sec == 365.25 * 24 * 3600
    @test problem.redshift_integral_fiducial == 1.0
    @test problem.strategy isa FullBNS
    @test redshift(problem) ≈ [0.1, 0.2]
end

@testset "load_cache rejects unsupported proposal_samples source_type" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "importance_context_julia.h5")
    path = joinpath(mktempdir(), "bad_source_type.h5")
    cp(fixture_path, path; force=true)
    h5open(path, "r+") do f
        g = f["proposal_samples"]
        delete_attribute(g, PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR)
        attributes(g)[PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR] = "BBH"
    end
    @test_throws ArgumentError load_cache(path, _TEST_LOAD_DETS)
end

@testset "importance_sampling_problem 5-arg infers redshift integral" begin
    fid = ProposalFiducialParameters(;
        H0=67.0,
        Omega_m=0.315,
        chi0=1.0,
        chin=0.0,
        gamma=2.7,
        kappa=3.0,
        z_peak=2.5,
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    ri = fiducial_redshift_integral(fid, spec)
    samples = FullBNSSamples(
        [1.4],
        [1.2],
        [0.1],
        [0.0],
        [0.0],
        [100.0],
        [100.0],
    )
    lp = reconstruct_proposal_log_prob(samples, spec, fid)
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        lp,
        reshape([1.4, 1.2, 0.1, 0.0, 0.0, 100.0, 100.0], 1, :),
        fill(1.0, 1, 2),
        [1.0],
    )
    observation = ObservationConfig(
        [1.0, 2.0],
        [1.0, 1.0],
        [1.0, 1.0],
        BitVector([true, true]),
        [0.0, 0.0],
        1.0,
        1.0,
    )
    p = importance_sampling_problem(proposal, observation, spec, 1.0, fid)
    @test p.redshift_integral_fiducial ≈ ri rtol = 1e-10
end
