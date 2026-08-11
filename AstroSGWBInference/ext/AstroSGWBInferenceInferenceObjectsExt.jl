"""
    AstroSGWBInferenceInferenceObjectsExt

netCDF-write helpers for `AstroSGWBInference`, loaded when `InferenceObjects` is.

Both functions are a few lines because `InferenceObjects.Dataset` is a
`DimensionalData.AbstractDimStack` and `Base.setindex(::InferenceData, ::Dataset, :posterior)`
already exists: rebuilding a group is rebuilding its layers.
"""
module AstroSGWBInferenceInferenceObjectsExt

using AstroSGWBInference: AstroSGWBInference, NETCDF_PARAMETER_NAMES
using InferenceObjects: InferenceObjects, Dataset, InferenceData

const DD = InferenceObjects.DimensionalData

function AstroSGWBInference.rename_posterior_for_netcdf(idata::InferenceData)
    haskey(idata, :posterior) || return idata
    posterior = idata.posterior
    layers = DD.layers(posterior)
    new_names = map(k -> get(NETCDF_PARAMETER_NAMES, k, k), keys(layers))
    # Two Unicode names colliding onto one ASCII name would silently drop a variable
    # (`DimStack` keys on the array names), so it is an error rather than a surprise in
    # the written file.
    allunique(new_names) || throw(ArgumentError(
        "NETCDF_PARAMETER_NAMES maps $(keys(layers)) onto colliding names $(new_names)",
    ))
    arrays = map((array, name) -> DD.rebuild(array; name = name),
        Tuple(layers), new_names)
    return Base.setindex(idata, Dataset(arrays...; metadata = DD.metadata(posterior)),
        :posterior)
end

function AstroSGWBInference.merge_into_posterior(idata::InferenceData, nt::NamedTuple)
    isempty(nt) && return idata
    posterior = idata.posterior
    # Reuse the posterior's own `(draw, chain)` dimensions rather than letting
    # `convert_to_dataset` generate fresh ones: identical dimension *names* with
    # independently-generated lookups is exactly the misalignment that would go unnoticed.
    sample_dims = DD.dims(posterior, (:draw, :chain))
    added = map(keys(nt), Tuple(nt)) do name, value
        size(value) == map(length, sample_dims) || throw(DimensionMismatch(
            "$(name) has size $(size(value)) but the posterior's (draw, chain) is " *
            "$(map(length, sample_dims))",
        ))
        return DD.DimArray(collect(value), sample_dims; name = name)
    end
    layers = merge(DD.layers(posterior), NamedTuple{keys(nt)}(added))
    return Base.setindex(
        idata, Dataset(values(layers)...; metadata = DD.metadata(posterior)), :posterior)
end

end
