"""
    GWBackgroundInferenceFlexiChainsExt

netCDF-write helpers for `GWBackgroundInference`, loaded when `FlexiChains` is.

Merge reuses `Base.merge` on two same-sized `FlexiChain`s; rename is
`FlexiChains.map_parameters` through [`NETCDF_PARAMETER_NAMES`](@ref). Callers
convert the renamed chain to `InferenceData` and write with `to_netcdf`.
"""
module GWBackgroundInferenceFlexiChainsExt

using GWBackgroundInference: GWBackgroundInference, NETCDF_PARAMETER_NAMES
using FlexiChains: FlexiChains, FlexiChain, Parameter, VarName

"""Build a `Parameter` key matching `chain`'s key type from a `Symbol` name."""
function _parameter_key(::FlexiChain{TKey}, name::Symbol) where {TKey}
    if TKey <: VarName
        return Parameter(VarName{name}())
    elseif TKey <: Symbol
        return Parameter(name)
    else
        throw(ArgumentError(
            "merge_into_posterior only supports FlexiChain{Symbol} and " *
            "FlexiChain{<:VarName}; got FlexiChain{$TKey}",
        ))
    end
end

function GWBackgroundInference.merge_into_posterior(
        chain::FlexiChain{TKey},
        nt::NamedTuple
) where {TKey}
    isempty(nt) && return chain
    niters, nchains = size(chain)
    for (name, value) in pairs(nt)
        size(value) == (niters, nchains) || throw(DimensionMismatch(
            "$(name) has size $(size(value)) but the chain's (iter, chain) is " *
            "$((niters, nchains))",
        ))
    end
    data = Dict(
        _parameter_key(chain, name) => collect(value) for (name, value) in pairs(nt)
    )
    added = FlexiChain{TKey}(
        niters,
        nchains,
        data;
        iter_indices = FlexiChains.iter_indices(chain),
        chain_indices = FlexiChains.chain_indices(chain)
    )
    return merge(chain, added)
end

function GWBackgroundInference.rename_posterior_for_netcdf(chain::FlexiChain)
    return FlexiChains.map_parameters(chain) do p
        get(NETCDF_PARAMETER_NAMES, Symbol(p), Symbol(p))
    end
end

end
