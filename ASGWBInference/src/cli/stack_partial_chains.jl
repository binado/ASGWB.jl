module StackPartialChainsCLI

using ASGWB
using ..ChainIO: atomic_save_chain
using AbstractMCMC: chainsstack
using MCMCChains: Chains
using JLD2
using Turing

function _glob_regex(pattern::AbstractString)
    io = IOBuffer()
    print(io, '^')
    for c in pattern
        if c == '*'
            print(io, ".*")
        elseif c == '?'
            print(io, '.')
        elseif c in ('.', '+', '(', ')', '[', ']', '{', '}', '^', '$', '|', '\\')
            print(io, '\\', c)
        else
            print(io, c)
        end
    end
    print(io, '$')
    return Regex(String(take!(io)))
end

function _expand_input(input::AbstractString)
    if !occursin('*', input) && !occursin('?', input)
        isfile(input) || throw(ArgumentError("input file not found: $input"))
        return [String(input)]
    end

    dir = dirname(input)
    dir = isempty(dir) ? "." : dir
    pattern = basename(input)
    isdir(dir) || throw(ArgumentError("glob directory not found: $dir"))

    re = _glob_regex(pattern)
    matches = sort([joinpath(dir, name) for name in readdir(dir) if occursin(re, name)])
    isempty(matches) && throw(ArgumentError("glob matched no files: $input"))
    return matches
end

function _read_chain(path::AbstractString)
    data = load(path)
    chain = if haskey(data, "chain")
        data["chain"]
    elseif haskey(data, "snapshot")
        data["snapshot"]
    else
        throw(ArgumentError("JLD2 file contains neither 'chain' nor 'snapshot' key: $path"))
    end
    chain isa Chains || throw(ArgumentError("not an MCMCChains.Chains file: $path"))
    size(chain, 3) == 1 || @warn "input has more than one chain" path size=size(chain)
    return chain
end

"""
Stack JLD2-saved MCMCChains files, such as per-chain checkpoint partials, into
one combined MCMCChains.Chains object.  Accepts both `chain` (final output)
and `snapshot` (checkpoint) keys.

# Args

- `inputs`: input chain files or quoted glob patterns, for example
  `"chains.partial.chain*.jld2"`.

# Options

- `-o, --output=<path>`: path for the stacked `.jld2` output.

- `-f, --force`: overwrite an existing output file.

Invoke from Julia, for example:

    using ASGWBInference
    ASGWBInference.StackPartialChainsCLI.stack("partials*.jld2"; output = "stacked.jld2")
"""
function stack(
        inputs::String...;
        output::String,
        force::Bool = false
)
    isempty(inputs) && throw(ArgumentError("provide at least one input file or glob"))
    if isfile(output) && !force
        throw(ArgumentError("output already exists, pass --force to overwrite: $output"))
    end

    paths = sort(unique(reduce(vcat, _expand_input.(inputs))))
    @info "reading chain files" count=length(paths)
    chains = [_read_chain(path) for path in paths]

    stacked = chainsstack(chains)
    output_dir = dirname(output)
    isempty(output_dir) || mkpath(output_dir)
    atomic_save_chain(output, stacked)

    @info "wrote stacked chain" path=output size=size(stacked)
    return nothing
end

end # module StackPartialChainsCLI
