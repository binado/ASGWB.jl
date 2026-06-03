# Test-only reference population implementing the three-method PopulationModel
# contract. The framework owns no concrete population types; callers define the
# concrete model used by their notebooks or scripts.
using ASGWB: PopulationModel, AbstractCosmology, OrderedUniformSourceMassPair,
             AlignedSpinChiSimple, redshift_prior, MadauDickinsonSourceFrame,
             BNS_LAMBDA_HIGH
import ASGWB: hyperparameters, hyperprior, single_event_prior
using Distributions: Uniform, product_distribution

struct ParityBNSPopulation <: PopulationModel end

hyperparameters(::ParityBNSPopulation) = (:γ, :κ, :zpeak)

function hyperprior(::ParityBNSPopulation)
    return product_distribution((
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    ))
end

function single_event_prior(::ParityBNSPopulation, cosmo::AbstractCosmology, Λ::NamedTuple)
    z_d = redshift_prior(MadauDickinsonSourceFrame(), cosmo, Λ)
    spin = AlignedSpinChiSimple()
    return product_distribution((
        mass = OrderedUniformSourceMassPair(),
        redshift = z_d,
        χ₁ = spin,
        χ₂ = spin,
        Λ₁ = Uniform(0.0, BNS_LAMBDA_HIGH),
        Λ₂ = Uniform(0.0, BNS_LAMBDA_HIGH)
    ))
end
