using Test
using CBCDistributions
using ASGWB

@testset "sample interpolation helpers" begin
    theta = canonical_hyperparameters(
        madau_dickinson_physical_model(),
        (;
            H0 = 67.0,
            Ωm = 0.315,
            Ξ₀ = 1.0,
            Ξₙ = 0.0,
            γ = 2.7,
            κ = 3.0,
            zpeak = 2.5
        )
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.0, 2.0, 101, nothing)
    z_grid = redshift_grid(spec)
    cosmology_cache,
    redshift_prior = cosmology_and_redshift_prior(
        cosmology(madau_dickinson_physical_model(), theta), theta, spec, z_grid
    )
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
