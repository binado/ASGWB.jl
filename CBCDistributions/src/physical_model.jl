using Distributions
using Distributions: ProductNamedTupleDistribution

abstract type ParametrizedModel end

Base.broadcastable(m::ParametrizedModel) = Ref(m)

struct PhysicalModel{C <: AbstractCosmology, P <: ParametrizedModel} <: ParametrizedModel
    cosmology_type::Type{C}
    population::P
end

function hyperparameters(model::PhysicalModel)
    return (hyperparameters(model.cosmology_type)..., hyperparameters(model.population)...)
end

function cosmology(model::PhysicalModel, Λ::NamedTuple)
    return cosmology(model.cosmology_type, Λ)
end

function population_prior(model::PhysicalModel, Λ::NamedTuple; kwargs...)
    return population_prior(model.population, cosmology(model, Λ), Λ; kwargs...)
end

struct PureModel{D <: Distribution} <: ParametrizedModel
    distribution::D
end

hyperparameters(::PureModel) = ()
function population_prior(model::PureModel, cosmology::AbstractCosmology, Λ::NamedTuple; kwargs...)
    model.distribution
end

struct PopulationModel{Names, Components} <: ParametrizedModel
    components::NamedTuple{Names, Components}

    function PopulationModel(components::NamedTuple{
            Names, Components}) where {Names, Components}
        wrapped = map(_as_population_component, components)
        flat = Symbol[]
        for component in values(wrapped)
            append!(flat, hyperparameters(component))
        end
        length(unique(flat)) == length(flat) ||
            throw(ArgumentError("population components define duplicate hyperparameters: $(Tuple(flat))"))
        WrappedComponents = typeof(wrapped).parameters[2]
        return new{Names, WrappedComponents}(wrapped)
    end
end

_as_population_component(component::ParametrizedModel) = component
_as_population_component(component::Distribution) = PureModel(component)

function hyperparameters(model::PopulationModel)
    return Tuple(Iterators.flatten(hyperparameters(c) for c in values(model.components)))
end

function population_prior(
        model::PopulationModel,
        cosmology::AbstractCosmology,
        Λ::NamedTuple;
        kwargs...
)
    return product_distribution((;
        (name => population_prior(component, cosmology, Λ; kwargs...)
    for (name, component) in pairs(model.components))...))
end

struct MadauDickinsonSourceFrameModel <: ParametrizedModel end

hyperparameters(::MadauDickinsonSourceFrameModel) = (:γ, :κ, :zpeak)

function population_prior(
        ::MadauDickinsonSourceFrameModel,
        cosmology::AbstractCosmology,
        Λ::NamedTuple;
        z_grid
)
    cache = CosmologyCache(cosmology, z_grid)
    γ, κ, zpeak = Λ.γ, Λ.κ, Λ.zpeak
    source_frame = z -> madau_dickinson_source_frame_distribution(z; γ, κ, zpeak)
    return RedshiftInterpolatedDistribution(build_redshift_prior(source_frame, cache))
end

function _component_batch_length(d::Distribution, samples::NamedTuple, key)
    haskey(samples, key) ||
        throw(ArgumentError("samples are missing population prior field $(repr(key))"))
    return _component_batch_length(d, samples[key], key)
end

function _batched_output_eltype(dists)
    isempty(dists) && return Float64
    return promote_type(map(eltype, values(dists))...)
end

function batched_logpdf(d::ProductNamedTupleDistribution, samples::NamedTuple)
    first_key = first(keys(d.dists))
    n = _component_batch_length(d.dists[first_key], samples, first_key)
    T = _batched_output_eltype(d.dists)
    out = zeros(T, n)
    for key in keys(d.dists)
        n_key = _component_batch_length(d.dists[key], samples, key)
        n_key == n ||
            throw(ArgumentError("population prior sample fields must have matching lengths"))
        _add_component_logpdf!(out, d.dists[key], samples[key])
    end
    return out
end

function batched_logpdf(d::Distribution, samples)
    return logpdf(d, samples)
end
