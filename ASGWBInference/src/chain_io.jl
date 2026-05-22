module ChainIO

using JLD2: jldsave
using MCMCChains: Chains, setinfo

"""
    slim_chain(chain)

Return a portable analysis artifact by dropping Turing resume state from
`chain.info`. Parameters and sampler internals remain in the chain value array.
"""
function slim_chain(chain::Chains)
    return setinfo(chain, NamedTuple())
end

function atomic_save_chain(path::AbstractString, chain::Chains)
    tmp = path * ".tmp"
    jldsave(tmp; chain = slim_chain(chain))
    mv(tmp, path; force = true)
    return nothing
end

end # module ChainIO
