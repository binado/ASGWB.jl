using Test
using InferenceObjects: InferenceObjects, convert_to_inference_data
using AstroSGWBInference: rename_posterior_for_netcdf, merge_into_posterior,
                          NETCDF_PARAMETER_NAMES

const DD = InferenceObjects.DimensionalData

# A (draw, chain) posterior with one Unicode name, one ASCII name, and one `:=` site --
# the three cases the rename has to distinguish.
function _example_idata(; ndraws = 5, nchains = 2)
    return convert_to_inference_data((
        H0 = randn(ndraws, nchains) .+ 67.0,
        Ωm = rand(ndraws, nchains),
        total_merger_rate = rand(ndraws, nchains)
    ))
end

@testset "rename_posterior_for_netcdf" begin
    idata = _example_idata()
    renamed = rename_posterior_for_netcdf(idata)

    @test Set(keys(DD.layers(renamed.posterior))) == Set((:H0, :Omega_m,
        :total_merger_rate))
    # ASCII names fall through untouched; only the mapped ones move.
    @test collect(renamed.posterior.H0) == collect(idata.posterior.H0)
    @test collect(renamed.posterior.Omega_m) == collect(idata.posterior.Ωm)
    @test collect(renamed.posterior.total_merger_rate) ==
          collect(idata.posterior.total_merger_rate)
    # The DimArrays' own `name` must move too, or `Dataset` keys and array names disagree
    # and the written file is inconsistent with itself.
    @test DD.name(renamed.posterior.Omega_m) === :Omega_m
    # Dimensions and other groups are untouched.
    @test DD.dims(renamed.posterior, :draw) == DD.dims(idata.posterior, :draw)
    @test keys(renamed) == keys(idata)

    # Idempotent: the ASCII names it produces map to themselves.
    @test Set(keys(DD.layers(rename_posterior_for_netcdf(renamed).posterior))) ==
          Set(keys(DD.layers(renamed.posterior)))
end

@testset "merge_into_posterior" begin
    idata = _example_idata(; ndraws = 5, nchains = 2)
    added = (
        H0_reconstructed = fill(70.0, 5, 2), quadrature_effective_nodes = fill(
            900.0, 5, 2))
    merged = merge_into_posterior(idata, added)

    @test Set(keys(DD.layers(merged.posterior))) ==
          Set((:H0, :Ωm, :total_merger_rate, :H0_reconstructed,
        :quadrature_effective_nodes))
    @test collect(merged.posterior.H0_reconstructed) == fill(70.0, 5, 2)
    # The new arrays reuse the posterior's own dimensions, so they align with the sampled
    # ones by construction rather than by two independently-generated lookups agreeing.
    @test DD.dims(merged.posterior.H0_reconstructed) == DD.dims(idata.posterior.H0)
    @test collect(merged.posterior.H0) == collect(idata.posterior.H0)

    @test merge_into_posterior(idata, NamedTuple()) === idata
    @test_throws DimensionMismatch merge_into_posterior(idata, (; bad = fill(1.0, 3, 2)))

    # And the two compose in the order `run_mcmc.jl` uses them.
    written = rename_posterior_for_netcdf(merged)
    @test haskey(DD.layers(written.posterior), :Omega_m)
    @test haskey(DD.layers(written.posterior), :quadrature_effective_nodes)
end
