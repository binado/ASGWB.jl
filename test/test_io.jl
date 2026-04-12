using HDF5
using Test

@testset "load_cache format v3 omits proposal_log_prob and dgw_fid_sq" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "importance_context_julia.h5")
    ref = load_cache(fixture_path)
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
    raw_flux = ref.proposal.cached_flux_over_dgw2 ./ reshape(scale, :, 1)
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
    expected_lp = log_prob_from_bundle.(z, Ref(bundle))

    path, io = mktemp()
    close(io)
    try
        h5open(path, "w") do f
            a = attributes(f)
            a["format_name"] = "asgwb.julia.importance_cache"
            a["format_version"] = 3
            a["local_merger_rate"] = ref.local_merger_rate
            a["observation_time_sec"] = ref.observation.observation_time_sec
            a["observation_time_yr"] = ref.observation.observation_time_yr
            a["redshift_integral_fiducial"] = ref.redshift_integral_fiducial

            write(f, "intrinsic_site_order", ref.proposal.intrinsic_site_order)
            write(
                f,
                "proposal_intrinsic_vector",
                Matrix(permutedims(ref.proposal.intrinsic_vector)),
            )
            write(f, "frequencies", ref.observation.frequencies)
            write(f, "in_band_mask", Vector{Bool}(ref.observation.in_band_mask))
            write(f, "fiducial_spectral_density", ref.observation.fiducial_spectral_density)
            write(f, "covariance", ref.observation.covariance)
            write(f, "sgwb_scale", ref.observation.sgwb_scale)
            write(f, "cached_flux", Matrix(permutedims(raw_flux)))

            g = create_group(f, "proposal_samples")
            write(g, "redshift", z)

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

        p = load_cache(path)
        @test p.proposal.cached_flux_over_dgw2 ≈ ref.proposal.cached_flux_over_dgw2
        @test p.proposal.dgw_fid_sq ≈ d_gw .^ 2
        @test p.proposal.log_prob ≈ expected_lp
    finally
        rm(path; force=true)
    end
end

@testset "importance_sampling_problem matches load_cache fixture" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "importance_context_julia.h5")
    from_file = load_cache(fixture_path)

    sgwb_scale = [1 / sqrt(63115200.0), 1 / sqrt(63115200.0)]
    proposal = ProposalData(
        ["redshift"],
        RedshiftOnlySamples([0.1, 0.2]),
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
    fid = ProposalFiducialParameters(; H0=67.0, Omega_m=0.315, chi0=1.0, chin=0.0)
    from_memory = importance_sampling_problem(
        proposal,
        observation,
        spec,
        161.0,
        1.0,
        fid,
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
    problem = load_cache(fixture_path)

    @test problem.proposal.intrinsic_site_order == ["redshift"]
    @test problem.proposal.samples.redshift ≈ [0.1, 0.2]
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
    @test problem.strategy isa RedshiftOnly
    @test redshift(problem) ≈ [0.1, 0.2]
end
