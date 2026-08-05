using Test
using AstroSGWBInference: MCMCConfig, SamplerConfig, load_config, save_config

function example_config(; sample_only = [:H0])
    sampler = SamplerConfig(3000, 3000, 0.9, "ForwardDiff", 0)
    fiducials = Dict{Symbol, Float64}(
        :H0 => 67.66,
        :Ωm => 0.3096,
        :w0 => -1.0,
        :Ξ₀ => 1.0,
        :Ξₙ => 1.91,
        :γ => 2.7,
        :κ => 3.0,
        :zpeak => 2.0,
        :R₀ => 161.0
    )
    return MCMCConfig(
        2,
        "catalog.h5",
        ["S1", "R1", "C1"],
        42,
        1.0,
        sampler,
        fiducials,
        sample_only,
        "chains",
        "chains"
    )
end

# A `MCMCConfig(d)` dict that mirrors `example_config`, used to exercise the
# validating constructor directly.
function example_dict()
    return Dict{String, Any}(
        "version" => 2,
        "catalog_path" => "catalog.h5",
        "detectors" => ["S1", "R1", "C1"],
        "seed" => 42,
        "observation_time" => 1.0,
        "sample_only" => ["H0"],
        "output_dir" => "chains",
        "output_prefix" => "chains",
        "sampler" => Dict{String, Any}(
            "nsamples" => 3000,
            "nadapts" => 3000,
            "target_acceptance" => 0.9,
            "ad_backend" => "ForwardDiff",
            "nchains" => 0
        ),
        "fiducials" => Dict{String, Any}(
            "H0" => 67.66,
            "Ωm" => 0.3096,
            "w0" => -1.0,
            "Ξ₀" => 1.0,
            "Ξₙ" => 1.91,
            "γ" => 2.7,
            "κ" => 3.0,
            "zpeak" => 2.0,
            "R₀" => 161.0
        )
    )
end

@testset "MCMCConfig round-trip" begin
    cfg = example_config()
    mktempdir() do dir
        path = joinpath(dir, "run.toml")
        save_config(cfg, path)

        @test isfile(path)
        @test !isfile(path * ".tmp")

        loaded = load_config(path)
        @test loaded == cfg
        @test loaded isa MCMCConfig
    end
end

@testset "constructor parity (dict vs struct)" begin
    @test MCMCConfig(example_dict()) == example_config()
end

@testset "Unicode fiducial keys survive round-trip" begin
    cfg = example_config()
    mktempdir() do dir
        path = joinpath(dir, "run.toml")
        save_config(cfg, path)

        # The file must contain quoted Unicode keys, not mangled ASCII.
        contents = read(path, String)
        @test occursin("\"Ξ₀\"", contents)
        @test occursin("\"Ωm\"", contents)

        loaded = load_config(path)
        for k in (:Ωm, :Ξ₀, :Ξₙ, :γ, :κ)
            @test loaded.fiducials[k] == cfg.fiducials[k]
        end
    end
end

@testset "sample_only: nothing is omitted and decoded" begin
    cfg = example_config(; sample_only = nothing)
    mktempdir() do dir
        path = joinpath(dir, "run.toml")
        save_config(cfg, path)

        contents = read(path, String)
        @test !occursin("sample_only", contents)

        loaded = load_config(path)
        @test loaded.sample_only === nothing
        @test loaded == cfg
    end

    # Set values must round-trip too.
    cfg2 = example_config(; sample_only = [:H0, :Ωm])
    mktempdir() do dir
        path = joinpath(dir, "run.toml")
        save_config(cfg2, path)
        loaded = load_config(path)
        @test loaded.sample_only == [:H0, :Ωm]
        @test loaded == cfg2
    end
end

@testset "validation failures throw" begin
    bad_version = example_dict()
    bad_version["version"] = 3
    @test_throws ArgumentError MCMCConfig(bad_version)

    bad_backend = example_dict()
    bad_backend["sampler"]["ad_backend"] = "Zygote"
    @test_throws ArgumentError MCMCConfig(bad_backend)

    bad_target = example_dict()
    bad_target["sampler"]["target_acceptance"] = 1.5
    @test_throws ArgumentError MCMCConfig(bad_target)

    bad_obs = example_dict()
    bad_obs["observation_time"] = 0.0
    @test_throws ArgumentError MCMCConfig(bad_obs)
end
