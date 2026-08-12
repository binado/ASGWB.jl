using ForwardDiff
using Random
using Distributions: insupport, logpdf
using Trapezoid: trapz

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
    R₀ = 161.0
    model = MadauDickinsonSourceFrame(; γ, κ, zpeak, R₀)
    amp = 1.0e-9 * R₀ / JULIAN_YEAR_SEC
    @test source_frame_distribution(model, 1.0) ≈
          amp * madau_dickinson_source_frame_distribution(1.0; γ, κ, zpeak)
end

@testset "redshift prior from cosmology grid" begin
    Λ = (γ = 2.7, κ = 3.0, zpeak = 2.5, R₀ = 161.0)
    cosmo = LambdaCDM(67.0, 0.315)
    z_grid = collect(LinRange(0.0, 2.0, 101))
    source_model = MadauDickinsonSourceFrame(
        γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak, R₀ = Λ.R₀)

    # Volume-array constructor (cosmology-decoupled, typed source frame)
    grid = distance_and_volume_grid(cosmo, z_grid)
    distribution = RedshiftInterpolatedDistribution(
        source_model, grid.differential_comoving_volume, z_grid)
    expected = @. 4π * grid.differential_comoving_volume *
                  source_frame_distribution(source_model, z_grid) / (1 + z_grid)
    @test distribution.dist.x == z_grid
    @test distribution.dist.y ≈ expected
    @test normalizer(distribution) === trapz(distribution.dist.x, distribution.dist.y)

    # Shape-only integral × amplitude recovers the same normalizer
    shape = madau_dickinson_source_frame_distribution.(
        z_grid; γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak)
    shape_y = @. 4π * grid.differential_comoving_volume * shape / (1 + z_grid)
    @test normalizer(distribution) ≈
          (1.0e-9 * Λ.R₀ / JULIAN_YEAR_SEC) * trapz(z_grid, shape_y)

    # Wrap-inner constructor
    wrapped = RedshiftInterpolatedDistribution(Interpolated1DDistribution(z_grid, expected))
    @test normalizer(wrapped) ≈ normalizer(distribution)

    @test minimum(distribution) == first(z_grid)
    @test maximum(distribution) == last(z_grid)
    @test isfinite(logpdf(distribution, 0.5))
    @test logpdf(distribution, -0.1) == -Inf
    @test logpdf(distribution, 2.1) == -Inf

    samples = rand(MersenneTwister(1234), distribution, 100)
    @test all(x -> insupport(distribution, x), samples)
end

@testset "redshift prior preserves AD" begin
    Λ = (γ = 2.7, κ = 3.0, zpeak = 2.5, R₀ = 161.0)
    z_grid = collect(LinRange(0.0, 2.0, 101))
    f = Ωm -> begin
        source_model = MadauDickinsonSourceFrame(
            γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak, R₀ = Λ.R₀)
        grid = distance_and_volume_grid(LambdaCDM(67.0, Ωm), z_grid)
        distribution = RedshiftInterpolatedDistribution(
            source_model, grid.differential_comoving_volume, z_grid)
        normalizer(distribution)
    end
    derivative = ForwardDiff.derivative(f, 0.315)
    @test isfinite(derivative)
    @test derivative != 0.0
end
