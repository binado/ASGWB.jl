using ForwardDiff
using Random
using Distributions: insupport, logpdf

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
    model = MadauDickinsonSourceFrame(; γ, κ, zpeak)
    @test source_frame_distribution(model, 1.0) ≈
          madau_dickinson_source_frame_distribution(1.0; γ, κ, zpeak)
end

@testset "redshift prior from cosmology grid" begin
    Λ = (γ = 2.7, κ = 3.0, zpeak = 2.5)
    cosmo = LambdaCDM(67.0, 0.315)
    z_grid = collect(LinRange(0.0, 2.0, 101))
    source_model = MadauDickinsonSourceFrame(
        γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak)
    source_frame_fn = z -> source_frame_distribution(source_model, z)

    prior = build_redshift_prior(source_frame_fn, cosmo, z_grid)
    grid = distance_and_volume_grid(cosmo, z_grid)
    expected = detector_frame_merger_rate_density.(
        z_grid,
        grid.differential_comoving_volume,
        source_frame_fn.(z_grid)
    )
    @test prior.x == z_grid
    @test prior.y ≈ expected
    @test redshift_integral(prior) === trapz(prior.x, prior.y)

    distribution = redshift_prior(source_model, cosmo; z_grid)
    @test minimum(distribution) == first(z_grid)
    @test maximum(distribution) == last(z_grid)
    @test isfinite(logpdf(distribution, 0.5))
    @test logpdf(distribution, -0.1) == -Inf
    @test logpdf(distribution, 2.1) == -Inf

    samples = rand(MersenneTwister(1234), distribution, 100)
    @test all(x -> insupport(distribution, x), samples)
end

@testset "redshift prior preserves AD" begin
    Λ = (γ = 2.7, κ = 3.0, zpeak = 2.5)
    z_grid = collect(LinRange(0.0, 2.0, 101))
    f = Ωm -> begin
        source_model = MadauDickinsonSourceFrame(
            γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak)
        distribution = redshift_prior(source_model, LambdaCDM(67.0, Ωm); z_grid)
        redshift_integral(distribution.prior)
    end
    derivative = ForwardDiff.derivative(f, 0.315)
    @test isfinite(derivative)
    @test derivative != 0.0
end
