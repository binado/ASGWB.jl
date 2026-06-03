# Test-only hyperparameter priors (not part of the CBCDistributions package API).
using Distributions
using CBCDistributions

struct TestPop <: PopulationModel end

CBCDistributions.hyperparameters(::TestPop) = (:α, :β)

function population_hyperprior(::TestPop)
    return product_distribution((
        α = Uniform(0.0, 1.0),
        β = Uniform(1.0, 2.0)
    ))
end

function CBCDistributions.single_event_prior(
        ::TestPop,
        cosmo::AbstractCosmology,
        Λ::NamedTuple
)
    return product_distribution((x = Uniform(0.0, Λ.α), y = Uniform(0.0, Λ.β)))
end

function cosmology_hyperprior(::Type{LambdaCDM})
    return product_distribution((
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95)
    ))
end

function cosmology_hyperprior(::Type{W0CDM})
    return product_distribution(merge(
        cosmology_hyperprior(LambdaCDM).dists,
        (w0 = Uniform(-3.0, 0.0),)
    ))
end

function cosmology_hyperprior(::Type{W0WaCDM})
    return product_distribution(merge(
        cosmology_hyperprior(W0CDM).dists,
        (wa = Uniform(-3.0, 3.0),)
    ))
end

function cosmology_hyperprior(::Type{<:ModifiedPropagation{C}}) where {C <: AbstractCosmology}
    return product_distribution(merge(
        cosmology_hyperprior(C).dists,
        (Ξ₀ = Uniform(0.5, 5.0), Ξₙ = Uniform(0.05, 3.0))
    ))
end

function merge_hyperpriors(cosmo_hp, pop_hp)
    return product_distribution(merge(cosmo_hp.dists, pop_hp.dists))
end
