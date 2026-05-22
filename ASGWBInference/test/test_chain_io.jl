using Test
using JLD2
using MCMCChains
using ASGWBInference: ChainIO, StackPartialChainsCLI

@testset "chain I/O artifacts" begin
    chain = Chains(
        reshape(collect(1.0:12.0), 3, 2, 2),
        [:H0, :lp],
        Dict(:parameters => [:H0], :internals => [:lp]);
        info = (model = :model, sampler = :sampler, samplerstate = :state)
    )

    slim = ChainIO.slim_chain(chain)
    @test slim isa Chains
    @test names(slim, :parameters) == names(chain, :parameters)
    @test names(slim, :internals) == names(chain, :internals)
    @test Array(slim) == Array(chain)
    @test isempty(keys(slim.info))

    mktempdir() do dir
        output = joinpath(dir, "chain.jld2")
        ChainIO.atomic_save_chain(output, chain)

        @test isfile(output)
        @test !isfile(output * ".tmp")
        data = load(output)
        @test haskey(data, "chain")
        loaded = data["chain"]
        @test loaded isa Chains
        @test isempty(keys(loaded.info))
        @test names(loaded, :parameters) == [:H0]
        @test names(loaded, :internals) == [:lp]
    end
end

@testset "stacked partial chains are slim" begin
    mktempdir() do dir
        for i in 1:2
            snapshot = Chains(
                reshape(collect(1.0:6.0) .+ 10i, 3, 2, 1),
                [:H0, :lp],
                Dict(:parameters => [:H0], :internals => [:lp]);
                info = (model = :model, sampler = :sampler, samplerstate = :state)
            )
            jldsave(joinpath(dir, "partial.chain$i.jld2"); snapshot)
        end

        output = joinpath(dir, "stacked.jld2")
        StackPartialChainsCLI.stack(
            joinpath(dir, "partial.chain*.jld2"); output, force = true)

        loaded = load(output, "chain")
        @test loaded isa Chains
        @test size(loaded, 3) == 2
        @test isempty(keys(loaded.info))
        @test names(loaded, :parameters) == [:H0]
        @test names(loaded, :internals) == [:lp]
    end
end
