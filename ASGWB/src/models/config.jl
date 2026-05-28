using SHA: sha256
using TOML

const MADAU_DICKINSON_MODIFIED_PROPAGATION_CONFIG_NAME = "madau_dickinson_modified_propagation"

"""
    ModelConfig

Model-side configuration loaded from `model.toml`: the structural forward model,
its canonical fiducial hyperparameter `NamedTuple`, and the redshift prior grid/spec.
Observation settings are supplied by the caller when assembling an
[`ImportanceSamplingProblem`](@ref).
"""
struct ModelConfig{M <: AbstractASGWBModel}
    model::M
    fiducial_hyperparameters::NamedTuple
    redshift_prior_spec::RedshiftPriorSpec
end

"""
    model_sha256_of_file(path) -> String

SHA-256 hex digest of the raw bytes of `path`.
"""
function model_sha256_of_file(path::AbstractString)::String
    return bytes2hex(sha256(read(path)))
end

# --- Shared TOML helpers ---

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
        throw(ArgumentError("model.toml [redshift].time_delay_model must be a string"))
    stripped = strip(value)
    (isempty(stripped) || stripped == "none") && return nothing
    throw(ArgumentError("time_delay_model=$(repr(value)) is not implemented"))
end

function _external_values(table::AbstractDict, mapping::NamedTuple, table_name::AbstractString)
    return (;
        (k => _require_real(table, external, table_name)
    for (k, external) in pairs(mapping))...)
end

function _parameters_config_dict(model::AbstractASGWBModel, Λ::NamedTuple)
    mapping = external_parameter_names(model)
    dict = Dict{String, Any}()
    for (k, external) in pairs(mapping)
        dict[external] = Λ[k]
    end
    return dict
end

function _redshift_config_dict(spec::RedshiftPriorSpec)
    return Dict{String, Any}(
        "z_min" => spec.z_min,
        "z_max" => spec.z_max,
        "num_interp" => spec.num_interp,
        "time_delay_model" => spec.time_delay_model === nothing ? "none" :
                              spec.time_delay_model
    )
end

# --- Generic TOML I/O ---

"""
    model_hyperparameters(data, model::AbstractASGWBModel) -> NamedTuple

Build the canonical fiducial hyperparameter state for `model` from parsed `model.toml` data.
"""
function model_hyperparameters(data::AbstractDict, model::AbstractASGWBModel)
    table = _require_table(data, "parameters")
    raw = _external_values(table, external_parameter_names(model), "parameters")
    return canonical_hyperparameters(model, raw; context = "fiducial hyperparameters")
end

"""
    redshift_prior_spec(data, model::AbstractASGWBModel) -> RedshiftPriorSpec

Build the redshift prior spec for `model` from parsed `model.toml` data.
"""
function redshift_prior_spec(data::AbstractDict, model::AbstractASGWBModel)
    redshift_table = _require_table(data, "redshift")
    tdm = _parse_time_delay_model(get(redshift_table, "time_delay_model", "none"))
    return RedshiftPriorSpec(
        redshift_prior_family(model),
        _require_real(redshift_table, "z_min", "redshift"),
        _require_real(redshift_table, "z_max", "redshift"),
        Int(redshift_table["num_interp"]),
        tdm
    )
end

"""
    model_config_dict(config::ModelConfig) -> Dict

Serialize `config` to a nested dict matching the `model.toml` schema.
"""
function model_config_dict(config::ModelConfig{M}) where {M <: AbstractASGWBModel}
    model = config.model
    return Dict{String, Any}(
        "model" => model_section_dict(model),
        "parameters" => _parameters_config_dict(model, config.fiducial_hyperparameters),
        "redshift" => _redshift_config_dict(config.redshift_prior_spec)
    )
end

# --- Model parsing and file I/O ---

function _parse_model(table::AbstractDict)
    name = _require_string(table, "name", "model")
    name == MADAU_DICKINSON_MODIFIED_PROPAGATION_CONFIG_NAME || throw(
        ArgumentError(
        "unknown model $(repr(name)); expected $(repr(MADAU_DICKINSON_MODIFIED_PROPAGATION_CONFIG_NAME))",
    ),
    )
    C = cosmology_type(_require_string(table, "cosmology", "model"))
    return MadauDickinsonModifiedPropagation{C}()
end

"""
    load_model_config(path) -> ModelConfig

Parse a sectioned `model.toml` file into a [`ModelConfig`](@ref).
"""
function load_model_config(path::AbstractString)
    data = TOML.parsefile(path)
    model = _parse_model(_require_table(data, "model"))
    Λ = model_hyperparameters(data, model)
    spec = redshift_prior_spec(data, model)
    return ModelConfig(model, Λ, spec)
end

"""
    save_model_config(path, config::ModelConfig)

Write `config` to the canonical `model.toml` schema readable by
[`load_model_config`](@ref).
"""
function save_model_config(path::AbstractString, config::ModelConfig)
    open(path, "w") do io
        TOML.print(io, model_config_dict(config))
    end
    return nothing
end
