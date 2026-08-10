# Test-only reference population implementing the PopulationModel contract.
using AstroSGWB: DEFAULT_Z_GRID, OrderedUniformSourceMassPair, AlignedSpinChiSimple,
                 redshift_prior, MadauDickinsonSourceFrame
using CBCDistributions: PopulationModel, full_hyperparameters, single_event_prior
import Cosmology
import Cosmology: AbstractCosmology
import CBCDistributions: single_event_prior
using Distributions: Uniform, product_distribution

struct ParityBNSPopulation <: PopulationModel end

Cosmology.hyperparameters(::ParityBNSPopulation) = (:γ, :κ, :zpeak)

function parity_population_hyperprior()
    return product_distribution((
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    ))
end

# Population sampler contract (used by the population-injection workflow): the per-event
# intrinsic prior as a product distribution.
function single_event_prior(
        ::ParityBNSPopulation,
        cosmo::AbstractCosmology,
        Λ::NamedTuple;
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
)
    z_d = redshift_prior(MadauDickinsonSourceFrame(), cosmo, Λ; z_grid)
    spin = AlignedSpinChiSimple()
    return product_distribution((
        mass = OrderedUniformSourceMassPair(),
        redshift = z_d,
        χ₁ = spin,
        χ₂ = spin,
        Λ₁ = Uniform(0.0, 5000.0),
        Λ₂ = Uniform(0.0, 5000.0)
    ))
end
