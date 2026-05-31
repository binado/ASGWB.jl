using Distributions: Uniform, product_distribution

"""
    BNSPopulationModel <: PopulationModel

Full binary-neutron-star population model: Madau–Dickinson source-frame
redshift distribution plus uniform mass, spin, and tidal-deformability priors.
Implements the three-method [`PopulationModel`](@ref) contract.  This is the
production caller model; the framework owns no concrete population types.
"""
struct BNSPopulationModel <: PopulationModel end

hyperparameters(::BNSPopulationModel) = (:γ, :κ, :zpeak)

function hyperprior(::BNSPopulationModel)
    return product_distribution((
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    ))
end

function single_event_prior(::BNSPopulationModel, cosmo::AbstractCosmology, Λ::NamedTuple)
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

"""
    bns_samples_from_catalog(catalog_samples::NamedTuple) -> NamedTuple

Restructure raw waveform-catalog sample columns into the struct-of-arrays proposal layout
the full-BNS `single_event_prior` expects: stack the two source masses into a `2 × n`
matrix and rename the ASCII spin/tidal columns (`chi_1`, `lambda_1`, …) to their Unicode
prior keys (`χ₁`, `Λ₁`, …). This is population-specific and lives in the caller layer.
"""
function bns_samples_from_catalog(catalog_samples::NamedTuple)
    return (
        mass = stack_source_masses(
            catalog_samples.mass_1_source, catalog_samples.mass_2_source),
        redshift = copy(catalog_samples.redshift),
        χ₁ = copy(catalog_samples.chi_1),
        χ₂ = copy(catalog_samples.chi_2),
        Λ₁ = copy(catalog_samples.lambda_1),
        Λ₂ = copy(catalog_samples.lambda_2)
    )
end

"""
    POPULATION_REGISTRY

Maps `[model].population` names to concrete [`PopulationModel`](@ref) instances.
Passed into [`load_model_toml`](@ref) by the inference CLI.
"""
const POPULATION_REGISTRY = Dict{String, PopulationModel}("bns" => BNSPopulationModel())
