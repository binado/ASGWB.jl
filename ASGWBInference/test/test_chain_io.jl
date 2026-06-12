using Test
using JLD2
using FlexiChains
using FlexiChains: Extra, Parameter, VNChain, @varname
using ASGWBInference: ChainIO

function synthetic_chain(values; chain_indices = 1:size(values, 2),
        last_sampler_state = fill(missing, size(values, 2)))
    return VNChain(
        size(values, 1),
        size(values, 2),
        Dict(
            Parameter(@varname(H0)) => values[:, :, 1],
            Extra(:lp) => values[:, :, 2]
        );
        chain_indices,
        last_sampler_state
    )
end

@testset "chain I/O artifacts" begin
    raw = reshape(collect(1.0:12.0), 3, 2, 2)
    chain = synthetic_chain(raw)

    mktempdir() do dir
        output = joinpath(dir, "chain.jld2")
        ChainIO.atomic_save_chain(output, chain)

        @test isfile(output)
        @test !isfile(output * ".tmp")
        data = load(output)
        @test haskey(data, "chain")
        loaded = data["chain"]
        @test loaded isa VNChain
        @test FlexiChains.parameters(loaded) == [@varname(H0)]
        @test FlexiChains.extras(loaded) == [Extra(:lp)]
        @test Array(loaded[:H0]) == raw[:, :, 1]
        @test Array(loaded[:lp]) == raw[:, :, 2]
        @test all(ismissing, FlexiChains.last_sampler_state(loaded))
    end
end
