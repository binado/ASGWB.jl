using Test
using CBCDistributions
using AstroSGWB

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
    C, P = LambdaCDM, ModifiedPropagation
    pop = ParityBNSPopulation()
    order = full_hyperparameters(C, P, pop)
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
    interp = GridInterpolator(samples, z_grid)

    # Batched interpolation of a *physical* dN_dz grid agrees with the scalar verb.
    @test interp(redshift_prior_dist.dN_dz.y) ≈
          [interpolate(redshift_prior_dist.dN_dz, z) for z in samples]

    # Interpolating the tabulated distances is the inference hot path's `d_L`. It is
    # deliberately linear rather than the exact within-cell antiderivative
    # `luminosity_distance(z, cache)` uses, so agreement is loose at low z by design.
    grid = distance_and_volume_grid(cosmo, z_grid)
    @test interp(grid.luminosity_distance) ≈
          [luminosity_distance(z, cosmology_cache) for z in samples] rtol = 1e-3

    # The sampling path still routes through CosmologyCache + the exact antiderivative.
    @test [cdf(cosmology_cache.inv_E_integral, z) for z in samples] ≈
          [comoving_distance(z, cosmo) / cosmology_cache.d_h for z in samples] rtol = 1e-3

    @test_throws ArgumentError GridInterpolator([-0.1], z_grid; check_bounds = true)
    @test_throws ArgumentError GridInterpolator([2.1], z_grid; check_bounds = true)
end
