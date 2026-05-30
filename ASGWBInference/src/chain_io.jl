module ChainIO

using JLD2: jldsave
using FlexiChains: VNChain

function atomic_save_chain(path::AbstractString, chain::VNChain; metadata = nothing)
    tmp = path * ".tmp"
    rm(tmp; force = true)
    try
        if metadata === nothing
            jldsave(tmp; chain)
        else
            jldsave(tmp; chain, metadata)
        end
    catch err
        rm(tmp; force = true)
        rethrow(err)
    end
    mv(tmp, path; force = true)
    return nothing
end

end # module ChainIO
