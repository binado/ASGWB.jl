using Distributions: Uniform

function madau_dickinson_physical_model(
        ::Type{C} = ModifiedPropagation{LambdaCDM}
) where {C <: AbstractCosmology}
    return PhysicalModel(C, _full_bns_population(MadauDickinsonSourceFrameModel()))
end

function _full_bns_population(redshift_component)
    spin = AlignedSpinChiSimple()
    return PopulationModel((
        mass = OrderedUniformSourceMassPair(),
        redshift = redshift_component,
        χ₁ = spin,
        χ₂ = spin,
        Λ₁ = Uniform(0.0, BNS_LAMBDA_HIGH),
        Λ₂ = Uniform(0.0, BNS_LAMBDA_HIGH)
    ))
end

function model_section_dict(model::PhysicalModel)
    Dict{String, Any}("cosmology" => cosmology_config_name(model.cosmology_type))
end

function redshift_prior_spec_from_population(model::PhysicalModel, redshift_component)
    redshift_component isa MadauDickinsonSourceFrameModel ||
        throw(ArgumentError("unsupported redshift population component $(typeof(redshift_component))"))
    return MadauDickinson
end
