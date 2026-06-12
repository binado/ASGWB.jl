using Test
using CBCDistributions
using ASGWB

if !@isdefined ParityBNSPopulation
    include(joinpath(@__DIR__, "fixture_population.jl"))
end

function _madau_dickinson_with_denom_exp(z, γ, denom_exp, zpeak)
    one_plus_z = 1 + z
    return ((one_plus_z^γ) / (1 + (one_plus_z / (1 + zpeak))^denom_exp)) *
           (1 + (1 + zpeak)^(-denom_exp))
end

@testset "Madau–Dickinson κ reparametrization" begin
    γ, κ, zpeak = 2.7, 3.0, 2.0
    denom_exp = γ + κ
    z_samples = [0.0, 0.5, zpeak, 3.0]

    @test madau_dickinson_source_frame_distribution(0.0; γ, κ, zpeak) ≈ 1.0
    for z in z_samples
        @test madau_dickinson_source_frame_distribution(z; γ, κ, zpeak) ≈
              _madau_dickinson_with_denom_exp(z, γ, denom_exp, zpeak)
    end
    @test source_frame_distribution(
        MadauDickinsonSourceFrame(), 1.0, (; γ, κ, zpeak)) ≈
          madau_dickinson_source_frame_distribution(1.0; γ, κ, zpeak)
end

@testset "sample interpolation helpers" begin
    C = ModifiedPropagation{LambdaCDM}
    pop = ParityBNSPopulation()
    order = full_hyperparameters(C, pop)
    theta = canonical_hyperparameters(
        order,
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
    z_grid = collect(LinRange(0.0, 2.0, 101))
    cosmo = cosmology(C, theta)
    cosmology_cache = CosmologyCache(cosmo, z_grid)
    redshift_prior_dist = build_redshift_prior(
        z -> madau_dickinson_source_frame_distribution(z; γ = theta.γ, κ = theta.κ,
            zpeak = theta.zpeak),
        cosmology_cache
    )
    samples = [0.0, 0.137, 0.9, 2.0]
    interp = SampleInterpolant(samples, z_grid)

    @test [_interpolate_at_sample(redshift_prior_dist.dN_dz.y, interp, i)
           for
           i in eachindex(samples)] ≈
          [interpolate(redshift_prior_dist.dN_dz, z) for z in samples]
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
