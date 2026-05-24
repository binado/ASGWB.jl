using Test
using Bijectors
using Distributions: product_distribution, Uniform
using ASGWB:
             MadauDickinsonModifiedPropagation,
             W0CDM,
             W0WaCDM,
             canonical_hyperparameters,
             hyperparameters,
             validate_hyperparameters,
             validate_prior

@testset "Madau-Dickinson modified-propagation model contract" begin
    model = MadauDickinsonModifiedPropagation()
    @test hyperparameters(model) == (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)

    prior_a = product_distribution((
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.0, 1.0),
        Ξ₀ = Uniform(0.0, 2.0),
        Ξₙ = Uniform(-1.0, 1.0),
        γ = Uniform(0.0, 5.0),
        κ = Uniform(0.0, 10.0),
        zpeak = Uniform(0.0, 5.0)
    ))
    prior_b = product_distribution((
        zpeak = Uniform(0.0, 5.0),
        κ = Uniform(0.0, 10.0),
        Ξₙ = Uniform(-1.0, 1.0),
        Ξ₀ = Uniform(0.0, 2.0),
        γ = Uniform(0.0, 5.0),
        Ωm = Uniform(0.0, 1.0),
        H0 = Uniform(20.0, 140.0)
    ))

    @test validate_prior(model, prior_a) === nothing
    @test_throws ArgumentError validate_prior(model, prior_b)

    θ = canonical_hyperparameters(
        model,
        (;
            H0 = 70.0,
            Ωm = 0.3,
            Ξ₀ = 1.0,
            Ξₙ = 0.0,
            γ = 2.0,
            κ = 3.0,
            zpeak = 1.5
        )
    )
    θ_unordered = (;
        zpeak = θ.zpeak,
        κ = θ.κ,
        γ = θ.γ,
        Ξₙ = θ.Ξₙ,
        Ξ₀ = θ.Ξ₀,
        Ωm = θ.Ωm,
        H0 = θ.H0
    )

    @test validate_hyperparameters(model, θ_unordered) === nothing
    @test canonical_hyperparameters(model, θ_unordered) == θ
    @test canonical_hyperparameters(model, θ_unordered; eltype = BigFloat).H0 isa BigFloat
    @test_throws ArgumentError validate_hyperparameters(model, (; H0 = 70.0, Ωm = 0.3))
    @test_throws ArgumentError validate_hyperparameters(model, merge(θ, (; extra = 1.0)))
    @test collect(Bijectors.link(prior_a, θ)) isa Vector

    @test hyperparameters(MadauDickinsonModifiedPropagation{W0CDM}()) ==
          (:H0, :Ωm, :w0, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    @test hyperparameters(MadauDickinsonModifiedPropagation{W0WaCDM}()) ==
          (:H0, :Ωm, :w0, :wa, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
end
