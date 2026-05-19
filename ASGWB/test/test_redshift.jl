using HDF5
using Test
using CBCDistributions

@testset "redshift parity" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    h5open(fixture_path, "r") do file
        group = file["redshift_case"]
        theta = HyperParameters(;
            H0 = Float64(read(group["theta/H0"])),
            Ωm = Float64(read(group["theta/Omega_m"])),
            γ = Float64(read(group["theta/gamma"])),
            κ = Float64(read(group["theta/kappa"])),
            zpeak = Float64(read(group["theta/z_peak"]))
        )
        spec = RedshiftPriorSpec(
            parse_redshift_prior_family(String(read(group["spec/family"]))),
            Float64(read(group["spec/z_min"])),
            Float64(read(group["spec/z_max"])),
            Int(read(group["spec/num_interp"])),
            nothing
        )
        sample_z = vec(Float64.(read(group["sample_z"])))
        expected_log_prob = vec(Float64.(read(group["log_prob"])))
        expected_integral = Float64(read(group["redshift_integral"]))

        redshift_prior = build_redshift_prior(theta, spec)

        # Fixture expected values were computed against the Python trapezoid-based bundle
        # norm; Julia now uses composite Simpson, so the tolerances reflect the trapezoid
        # vs Simpson discretization gap rather than numerical precision.
        @test redshift_log_prob.(Ref(redshift_prior), sample_z) ≈ expected_log_prob rtol = 5e-3
        @test redshift_integral(redshift_prior) ≈ expected_integral rtol = 5e-3
    end
end

@testset "sample interpolation helpers" begin
    theta = HyperParameters(;
        H0 = 67.0,
        Ωm = 0.315,
        γ = 2.7,
        κ = 3.0,
        zpeak = 2.5
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.0, 2.0, 101, nothing)
    z_grid = redshift_grid(spec)
    cosmology_cache, redshift_prior = cosmology_and_redshift_prior(theta, spec, z_grid)
    samples = [0.0, 0.137, 0.9, 2.0]
    interp = SampleInterpolant(samples, z_grid)

    @test [_interpolate_at_sample(redshift_prior.dN_dz.y, interp, i)
           for
           i in eachindex(samples)] ≈
          [interpolate(redshift_prior.dN_dz, z) for z in samples]
    @test [_cdf_at_sample(
               cosmology_cache.inv_E_integral.cumulative,
               cosmology_cache.inv_E_integral.y,
               interp,
               z_grid,
               i
           ) for i in eachindex(samples)] ≈
          [cdf(cosmology_cache.inv_E_integral, z) for z in samples]
    @test [luminosity_distance_at_sample(
               cosmology_cache, interp, z_grid, samples, i)
           for i in eachindex(samples)] ≈
          [luminosity_distance(z, cosmology_cache) for z in samples]
    @test_throws ArgumentError SampleInterpolant([-0.1], z_grid)
    @test_throws ArgumentError SampleInterpolant([2.1], z_grid)
end
