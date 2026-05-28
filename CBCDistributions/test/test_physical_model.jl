using Test
using Distributions
using CBCDistributions

@testset "PhysicalModel hyperparameters and population priors" begin
    redshift_model = MadauDickinsonSourceFrameModel()
    population = PopulationModel((
        mass = OrderedUniformSourceMassPair(),
        redshift = redshift_model,
        χ₁ = Uniform(-1.0, 1.0)
    ))
    model = PhysicalModel(ModifiedPropagation{LambdaCDM}, population)

    @test hyperparameters(redshift_model) == (:γ, :κ, :zpeak)
    @test hyperparameters(population) == (:γ, :κ, :zpeak)
    @test hyperparameters(model) == (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)

    Λ = (H0 = 67.4, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.0)
    prior = population_prior(model, Λ; z_grid = collect(LinRange(0.001, 2.0, 64)))
    samples = (
        mass = [1.4 1.3; 1.2 1.1],
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.5]
    )
    lp = batched_logpdf(prior, samples)
    @test length(lp) == 2
    @test all(isfinite, lp)

    @test_throws ArgumentError PopulationModel((
        a = redshift_model,
        b = redshift_model
    ))
end
