using Test
using ASGWBInference: RunInferenceCLI
using ASGWBInference

@testset "inference config discovery" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))

    withenv("ASGWB_REPO_ROOT" => nothing, "MCMC_CONFIG_FILEPATH" => nothing) do
        cd(repo) do
            @test RunInferenceCLI.default_config_path() ==
                  joinpath(repo, "config", "run_inference.toml")
            @test RunInferenceCLI.resolve_config_path("") ==
                  joinpath(repo, "config", "run_inference.toml")
        end
    end

    withenv("ASGWB_REPO_ROOT" => repo, "MCMC_CONFIG_FILEPATH" => "config/run_inference_smoke_h0.toml") do
        @test RunInferenceCLI.resolve_config_path("") ==
              joinpath(repo, "config", "run_inference_smoke_h0.toml")
    end

    absolute_config = joinpath(repo, "config", "run_inference_smoke_h0.toml")
    withenv("ASGWB_REPO_ROOT" => repo, "MCMC_CONFIG_FILEPATH" => absolute_config) do
        @test RunInferenceCLI.resolve_config_path("") == absolute_config
    end

    mktempdir() do tmp
        withenv("ASGWB_REPO_ROOT" => tmp, "MCMC_CONFIG_FILEPATH" => "custom.toml") do
            @test RunInferenceCLI.resolve_config_path("") == joinpath(tmp, "custom.toml")
        end
    end

    mktempdir() do tmp
        withenv("ASGWB_REPO_ROOT" => nothing, "MCMC_CONFIG_FILEPATH" => nothing) do
            cd(tmp) do
                err = try
                    RunInferenceCLI.repo_root()
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("set ASGWB_REPO_ROOT", sprint(showerror, err))
            end
        end
    end
end

@testset "sample_only config parsing" begin
    @test RunInferenceCLI.parse_sample_only(Dict{String, Any}()) === nothing
    @test RunInferenceCLI.parse_sample_only(Dict{String, Any}("sample_only" => nothing)) ===
          nothing
    @test RunInferenceCLI.parse_sample_only(Dict{String, Any}("sample_only" => ["H0"])) ==
          (:H0,)
    @test RunInferenceCLI.parse_sample_only(
        Dict{String, Any}("sample_only" => ["H0", "Omega_m"])) == (:H0, :Omega_m)

    @test_throws ArgumentError RunInferenceCLI.parse_sample_only(
        Dict{String, Any}("sample_only" => "H0"))
    @test_throws ArgumentError RunInferenceCLI.parse_sample_only(
        Dict{String, Any}("sample_only" => [1]))
end

@testset "julia_main rejects ARGS" begin
    old_args = copy(ARGS)
    empty!(ARGS)
    push!(ARGS, "--config=config/run_inference.toml")

    code = Cint(-1)
    message = mktemp() do path, io
        code = redirect_stderr(io) do
            ASGWBInference.julia_main()
        end
        flush(io)
        read(path, String)
    end

    empty!(ARGS)
    append!(ARGS, old_args)

    @test code == Cint(2)
    @test occursin("does not accept command-line arguments", message)
end
