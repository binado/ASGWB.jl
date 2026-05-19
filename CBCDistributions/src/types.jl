export RedshiftPriorFamily, parse_redshift_prior_family, RedshiftPriorSpec, MadauDickinson,
       PowerLaw

"""
    RedshiftPriorFamily

Closed set of redshift population models supported by [`RedshiftPriorSpec`](@ref).
File-backed caches store snake-case strings; use [`parse_redshift_prior_family`](@ref) when reading.
"""
@enum RedshiftPriorFamily MadauDickinson PowerLaw

"""
    parse_redshift_prior_family(s::AbstractString) -> RedshiftPriorFamily

Parse the HDF5 / Python cache string for `redshift_prior_spec.family`.
"""
function parse_redshift_prior_family(s::AbstractString)
    s == "madau_dickinson" && return MadauDickinson
    s == "power_law" && return PowerLaw
    throw(ArgumentError("unsupported redshift prior family $(repr(s))"))
end

"""
    RedshiftPriorSpec

Redshift grid settings for [`build_redshift_grid_bundle`](@ref). `time_delay_model`
is reserved for future parity with the Python stack; unsupported values must be
empty or `nothing` at load time.
"""
struct RedshiftPriorSpec
    family::RedshiftPriorFamily
    z_min::Float64
    z_max::Float64
    num_interp::Int
    time_delay_model::Union{String, Nothing}
end
