module ChainIO

using JLD2: jldsave
using FlexiChains: VNChain

function atomic_save_chain(path::AbstractString, chain::VNChain)
    tmp = path * ".tmp"
    jldsave(tmp; chain)
    mv(tmp, path; force = true)
    return nothing
end

end # module ChainIO
