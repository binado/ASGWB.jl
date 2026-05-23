using Test
using JLD2
using FlexiChains
using FlexiChains: Extra, Parameter, VNChain, @varname
using ASGWBInference: ChainIO, StackPartialChainsCLI

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

@testset "stacked partial chains preserve FlexiChains data" begin
    mktempdir() do dir
        for i in 1:2
            snapshot = synthetic_chain(
                reshape(collect(1.0:6.0) .+ 10i, 3, 1, 2);
                chain_indices = [i],
                last_sampler_state = ["state$i"]
            )
            jldsave(joinpath(dir, "partial.chain$i.jld2"); snapshot)
        end

        output = joinpath(dir, "stacked.jld2")
        StackPartialChainsCLI.stack(
            joinpath(dir, "partial.chain*.jld2"); output, force = true)

        loaded = load(output, "chain")
        @test loaded isa VNChain
        @test size(loaded) == (3, 2)
        @test FlexiChains.parameters(loaded) == [@varname(H0)]
        @test FlexiChains.extras(loaded) == [Extra(:lp)]
        @test Array(loaded[:H0])[:, 1] == collect(11.0:13.0)
        @test Array(loaded[:H0])[:, 2] == collect(21.0:23.0)
        @test Array(loaded[:lp])[:, 1] == collect(14.0:16.0)
        @test Array(loaded[:lp])[:, 2] == collect(24.0:26.0)
        @test FlexiChains.last_sampler_state(loaded) == ["state1", "state2"]
    end
end

@testset "MCMCChains artifacts are rejected before stacking" begin
    mktempdir() do dir
        input = joinpath(dir, "legacy.jld2")
        output = joinpath(dir, "stacked.jld2")
        jldsave(input; chain = (legacy = :mcmcchains,))

        @test_throws ArgumentError StackPartialChainsCLI.stack(input; output, force = true)
    end
end
