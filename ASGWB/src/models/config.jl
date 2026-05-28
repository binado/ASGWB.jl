using SHA: sha256
using TOML

const MADAU_DICKINSON_SOURCE_FRAME_CONFIG_NAME = "madau_dickinson_source_frame"

struct ModelConfig{M <: PhysicalModel}
    model::M
    fiducial_hyperparameters::NamedTuple
    redshift_prior_spec::RedshiftPriorSpec
end

model_sha256_of_file(path::AbstractString)::String = bytes2hex(sha256(read(path)))

function _require_table(data::AbstractDict, key::AbstractString)
    value = get(data, key, nothing)
    value isa AbstractDict || throw(ArgumentError("model.toml requires [$key] table"))
    return value
end

function _require_string(table::AbstractDict, key::AbstractString, table_name::AbstractString)
    haskey(table, key) ||
        throw(ArgumentError("model.toml [$table_name] requires $(repr(key))"))
    value = table[key]
    value isa AbstractString ||
        throw(ArgumentError("model.toml [$table_name].$key must be a string"))
    return value
end

function _require_real(table::AbstractDict, key::AbstractString, table_name::AbstractString)
    haskey(table, key) ||
        throw(ArgumentError("model.toml [$table_name] requires $(repr(key))"))
    return Float64(table[key])
end

function _parse_time_delay_model(value)
    value === nothing && return nothing
    value isa AbstractString ||
        throw(ArgumentError("model.toml time_delay_model must be a string"))
    stripped = strip(value)
    (isempty(stripped) || stripped == "none") && return nothing
    throw(ArgumentError("time_delay_model=$(repr(value)) is not implemented"))
end

function _external_values(table::AbstractDict, mapping::NamedTuple, table_name::AbstractString)
    return (;
        (k => _require_real(table, external, table_name)
    for (k, external) in pairs(mapping))...)
end

function _parameters_config_dict(model::PhysicalModel, Λ::NamedTuple)
    mapping = external_parameter_names(model)
    dict = Dict{String, Any}()
    for (k, external) in pairs(mapping)
        dict[external] = Λ[k]
    end
    return dict
end

function _redshift_config_dict(spec::RedshiftPriorSpec)
    return Dict{String, Any}(
        "model" => MADAU_DICKINSON_SOURCE_FRAME_CONFIG_NAME,
        "z_min" => spec.z_min,
        "z_max" => spec.z_max,
        "num_interp" => spec.num_interp,
        "time_delay_model" => spec.time_delay_model === nothing ? "none" :
                              spec.time_delay_model
    )
end

function model_hyperparameters(data::AbstractDict, model::PhysicalModel)
    table = _require_table(data, "parameters")
    raw = _external_values(table, external_parameter_names(model), "parameters")
    return canonical_hyperparameters(model, raw; context = "fiducial hyperparameters")
end

function _redshift_population_component(data::AbstractDict)
    population = _require_table(data, "population")
    redshift = _require_table(population, "redshift")
    name = _require_string(redshift, "model", "population.redshift")
    name == MADAU_DICKINSON_SOURCE_FRAME_CONFIG_NAME ||
        throw(ArgumentError("unknown redshift population model $(repr(name))"))
    return redshift, MadauDickinsonSourceFrameModel()
end

function redshift_prior_spec(data::AbstractDict, model::PhysicalModel)
    redshift_table, component = _redshift_population_component(data)
    tdm = _parse_time_delay_model(get(redshift_table, "time_delay_model", "none"))
    family = redshift_prior_spec_from_population(model, component)
    return RedshiftPriorSpec(
        family,
        _require_real(redshift_table, "z_min", "population.redshift"),
        _require_real(redshift_table, "z_max", "population.redshift"),
        Int(redshift_table["num_interp"]),
        tdm
    )
end

function model_config_dict(config::ModelConfig)
    return Dict{String, Any}(
        "model" => model_section_dict(config.model),
        "population" => Dict{String, Any}(
            "redshift" => _redshift_config_dict(config.redshift_prior_spec)
        ),
        "parameters" => _parameters_config_dict(config.model, config.fiducial_hyperparameters)
    )
end

function _parse_model(data::AbstractDict)
    model_table = _require_table(data, "model")
    C = cosmology_type(_require_string(model_table, "cosmology", "model"))
    _, redshift_component = _redshift_population_component(data)
    return PhysicalModel(C, _full_bns_population(redshift_component))
end

function load_model_config(path::AbstractString)
    data = TOML.parsefile(path)
    model = _parse_model(data)
    Λ = model_hyperparameters(data, model)
    spec = redshift_prior_spec(data, model)
    return ModelConfig(model, Λ, spec)
end

function save_model_config(path::AbstractString, config::ModelConfig)
    open(path, "w") do io
        TOML.print(io, model_config_dict(config))
    end
    return nothing
end
