using Test
using FlexiChains
using FlexiChains: FlexiChain, Parameter
using GWBackgroundInference: rename_posterior_for_netcdf, merge_into_posterior,
                          NETCDF_PARAMETER_NAMES

# A (iter, chain) FlexiChain with one Unicode name, one ASCII name, and one `:=` site --
# the three cases the rename has to distinguish.
function _example_chain(; niters = 5, nchains = 2)
    return FlexiChain{Symbol}(
        niters,
        nchains,
        Dict(
            Parameter(:H0) => randn(niters, nchains) .+ 67.0,
            Parameter(:Ωm) => rand(niters, nchains),
            Parameter(:total_merger_rate) => rand(niters, nchains)
        )
    )
end

@testset "rename_posterior_for_netcdf" begin
    chain = _example_chain()
    renamed = rename_posterior_for_netcdf(chain)

    @test Set(Symbol.(FlexiChains.parameters(renamed))) ==
          Set((:H0, :Omega_m, :total_merger_rate))
    # ASCII names fall through untouched; only the mapped ones move.
    @test Array(renamed[Parameter(:H0)]) == Array(chain[Parameter(:H0)])
    @test Array(renamed[Parameter(:Omega_m)]) == Array(chain[Parameter(:Ωm)])
    @test Array(renamed[Parameter(:total_merger_rate)]) ==
          Array(chain[Parameter(:total_merger_rate)])

    # Idempotent: the ASCII names it produces map to themselves.
    @test Set(Symbol.(FlexiChains.parameters(rename_posterior_for_netcdf(renamed)))) ==
          Set(Symbol.(FlexiChains.parameters(renamed)))
end

@testset "merge_into_posterior" begin
    chain = _example_chain(; niters = 5, nchains = 2)
    added = (
        H0_reconstructed = fill(70.0, 5, 2),
        quadrature_effective_nodes = fill(900.0, 5, 2)
    )
    merged = merge_into_posterior(chain, added)

    @test Set(Symbol.(FlexiChains.parameters(merged))) == Set((
        :H0, :Ωm, :total_merger_rate, :H0_reconstructed, :quadrature_effective_nodes))
    @test Array(merged[Parameter(:H0_reconstructed)]) == fill(70.0, 5, 2)
    @test Array(merged[Parameter(:H0)]) == Array(chain[Parameter(:H0)])

    @test merge_into_posterior(chain, NamedTuple()) === chain
    @test_throws DimensionMismatch merge_into_posterior(chain, (; bad = fill(1.0, 3, 2)))

    # And the two compose in the order `run_mcmc.jl` uses them.
    written = rename_posterior_for_netcdf(merged)
    @test :Omega_m in Symbol.(FlexiChains.parameters(written))
    @test :quadrature_effective_nodes in Symbol.(FlexiChains.parameters(written))
end
