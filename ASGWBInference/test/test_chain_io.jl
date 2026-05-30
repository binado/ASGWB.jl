using Test
using JLD2
using FlexiChains
using FlexiChains: Extra, Parameter, VNChain, @varname
using ASGWBInference: ChainIO, RunInferenceCLI, StackPartialChainsCLI

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
        @test !haskey(data, "metadata")
        loaded = data["chain"]
        @test loaded isa VNChain
        @test FlexiChains.parameters(loaded) == [@varname(H0)]
        @test FlexiChains.extras(loaded) == [Extra(:lp)]
        @test Array(loaded[:H0]) == raw[:, :, 1]
        @test Array(loaded[:lp]) == raw[:, :, 2]
        @test all(ismissing, FlexiChains.last_sampler_state(loaded))
    end
end

@testset "chain I/O artifacts can include metadata" begin
    raw = reshape(collect(1.0:12.0), 3, 2, 2)
    chain = synthetic_chain(raw)
    metadata = Dict{String, Any}(
        "schema_version" => 1,
        "artifact_type" => "run_inference",
        "seed" => 7
    )

    mktempdir() do dir
        output = joinpath(dir, "chain-with-metadata.jld2")
        ChainIO.atomic_save_chain(output, chain; metadata)

        data = load(output)
        @test data["chain"] isa VNChain
        @test data["metadata"] == metadata
        @test !isfile(output * ".tmp")
    end
end

@testset "run metadata builder records curated fields" begin
    dets = [(name = "H1",), (name = "L1",)]
    cache_path = "/tmp/cache.h5"
    fiducial = (
        H0 = 67.0,
        Ωm = 0.315,
        Ξ₀ = 1.0,
        Ξₙ = 0.0,
        γ = 2.7,
        κ = 5.6,
        zpeak = 2.0,
        Λ = nothing,
        w0 = nothing,
        wa = nothing
    )
    problem = (fiducial_parameters = fiducial,)
    settings = Dict{String, Any}(
        "cache_path" => cache_path,
        "detectors" => ["H1", "L1"],
        "seed" => 11,
        "model" => Dict{String, Any}("cosmology" => "LambdaCDM"),
        "init" => Dict{String, Any}(
            "H0" => 67.66,
            "Ωm" => 0.3096,
            "Ξ₀" => 1.0,
            "Ξₙ" => 1.91,
            "γ" => 2.7,
            "κ" => 5.7,
            "zpeak" => 2.0
        )
    )
    model = RunInferenceCLI._resolve_inference_model(settings)
    selected_priors = RunInferenceCLI.select_priors(RunInferenceCLI.PRIORS, (:H0,))

    metadata = RunInferenceCLI.build_run_metadata(
        settings,
        cache_path,
        dets,
        model,
        problem,
        selected_priors,
        100,
        50,
        2,
        0.8,
        11
    )

    @test metadata["schema_version"] == 1
    @test metadata["artifact_type"] == "run_inference"
    @test metadata["cache_path"] == cache_path
    @test metadata["detectors"] == ["H1", "L1"]
    @test metadata["model"]["cosmology"] == "LambdaCDM"
    @test metadata["init"]["Ωm"] == 0.3096
    @test metadata["fiducial_parameters"]["H0"] == fiducial.H0
    @test haskey(metadata["priors"], "H0")
    @test metadata["sampler"] == Dict{String, Any}(
        "n_samples" => 100,
        "n_adapts" => 50,
        "num_chains" => 2,
        "target_acceptance" => 0.8
    )
    @test metadata["seed"] == 11
    @test haskey(metadata["git"], "revision")
    @test haskey(metadata["versions"], "julia")
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

        data = load(output)
        @test !haskey(data, "metadata")
        loaded = data["chain"]
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
