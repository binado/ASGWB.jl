using Test
using Bijectors
using Distributions: product_distribution, Uniform
using ASGWB:
             ModifiedPropagation,
             LambdaCDM, W0CDM, W0WaCDM,
             full_hyperparameters,
             full_hyperprior,
             canonical_hyperparameters,
             validate_hyperparameters
using ASGWBInference: validate_hyperprior

if !@isdefined ParityBNSPopulation
    include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "fixture_population.jl"))
end

@testset "caller-defined population hyperparameter contract" begin
    C = ModifiedPropagation{LambdaCDM}
    pop = ParityBNSPopulation()
    order = full_hyperparameters(C, pop)

    @test order == (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)

    prior_a = full_hyperprior(C, pop)
    prior_b = product_distribution((
        zpeak = Uniform(0.0, 5.0),
        κ = Uniform(0.0, 10.0),
        Ξₙ = Uniform(-1.0, 1.0),
        Ξ₀ = Uniform(0.0, 2.0),
        γ = Uniform(0.0, 5.0),
        Ωm = Uniform(0.0, 1.0),
        H0 = Uniform(20.0, 140.0)
    ))

    @test validate_hyperprior(order, prior_a) === nothing
    @test_throws ArgumentError validate_hyperprior(order, prior_b)

    θ = canonical_hyperparameters(
        order,
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

    @test_throws ArgumentError validate_hyperparameters(order, θ_unordered)
    @test canonical_hyperparameters(order, θ_unordered) == θ
    @test canonical_hyperparameters(order, θ_unordered; eltype = BigFloat).H0 isa BigFloat
    @test_throws ArgumentError validate_hyperparameters(order, (; H0 = 70.0, Ωm = 0.3))
    @test_throws ArgumentError validate_hyperparameters(order, merge(θ, (; extra = 1.0)))
    @test collect(Bijectors.link(prior_a, θ)) isa Vector

    @test full_hyperparameters(ModifiedPropagation{W0CDM}, ParityBNSPopulation()) ==
          (:H0, :Ωm, :w0, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    @test full_hyperparameters(ModifiedPropagation{W0WaCDM}, ParityBNSPopulation()) ==
          (:H0, :Ωm, :w0, :wa, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
end
