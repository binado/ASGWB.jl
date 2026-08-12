using Test
using AstroSGWBInference: MCMCConfig, SamplerConfig, load_config, save_config,
                          posterior_params, NETCDF_PARAMETER_NAMES

function example_config(; sample_only = [:H0], likelihood = "default",
        amplitude_parameter = nothing, amplitude_num_nodes = 1024,
        amplitude_prior_span_sigma = 10.0)
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
        3,
        "catalog.h5",
        ["S1", "R1", "C1"],
        42,
        1.0,
        sampler,
        fiducials,
        sample_only,
        likelihood,
        amplitude_parameter,
        amplitude_num_nodes,
        amplitude_prior_span_sigma,
        "chains",
        "chains"
    )
end

# A `MCMCConfig(d)` dict that mirrors `example_config`, used to exercise the
# validating constructor directly.
function example_dict()
    return Dict{String, Any}(
        "version" => 3,
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
    bad_version["version"] = 2
    @test_throws ArgumentError MCMCConfig(bad_version)

    enzyme_backend = example_dict()
    enzyme_backend["sampler"]["ad_backend"] = "Enzyme"
    @test MCMCConfig(enzyme_backend).sampler.ad_backend == "Enzyme"

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

# --------------------------------------------------------------------------
# Schema v3: likelihood selection
# --------------------------------------------------------------------------

function marginalized_dict(; amplitude_parameter = "H0", sample_only = ["Ωm"])
    d = example_dict()
    d["likelihood"] = "amplitude_marginalized"
    d["amplitude_parameter"] = amplitude_parameter
    d["sample_only"] = sample_only
    return d
end

@testset "v3 defaults" begin
    # `likelihood` and the amplitude knobs are all optional with the documented defaults,
    # so a v2-shaped body with `version = 3` still loads.
    cfg = MCMCConfig(example_dict())
    @test cfg.likelihood == "default"
    @test cfg.amplitude_parameter === nothing
    @test cfg.amplitude_num_nodes == 1024
    @test cfg.amplitude_prior_span_sigma == 10.0
end

@testset "amplitude-marginalized config round-trip" begin
    cfg = example_config(; sample_only = [:Ωm], likelihood = "amplitude_marginalized",
        amplitude_parameter = :H0, amplitude_num_nodes = 512,
        amplitude_prior_span_sigma = 6.0)
    @test MCMCConfig(marginalized_dict()) ==
          example_config(; sample_only = [:Ωm], likelihood = "amplitude_marginalized",
        amplitude_parameter = :H0)

    mktempdir() do dir
        path = joinpath(dir, "run.toml")
        save_config(cfg, path)
        loaded = load_config(path)
        @test loaded == cfg
        @test loaded.likelihood == "amplitude_marginalized"
        @test loaded.amplitude_parameter === :H0
        @test loaded.amplitude_num_nodes == 512
        @test loaded.amplitude_prior_span_sigma == 6.0
    end

    # A Unicode amplitude parameter survives the quoted-key round trip too.
    unicode = example_config(; sample_only = [:H0], likelihood = "amplitude_marginalized",
        amplitude_parameter = :R₀)
    mktempdir() do dir
        path = joinpath(dir, "run.toml")
        save_config(unicode, path)
        @test occursin("\"R₀\"", read(path, String))
        @test load_config(path) == unicode
    end

    # `nothing` is omitted, as for `sample_only`.
    mktempdir() do dir
        path = joinpath(dir, "run.toml")
        save_config(example_config(), path)
        @test !occursin("amplitude_parameter", read(path, String))
    end
end

@testset "likelihood/amplitude_parameter coupling" begin
    missing_parameter = marginalized_dict()
    delete!(missing_parameter, "amplitude_parameter")
    @test_throws ArgumentError MCMCConfig(missing_parameter)

    # The other half: a default-likelihood run with an amplitude parameter would silently
    # ignore it, so it is rejected rather than tolerated.
    stray_parameter = example_dict()
    stray_parameter["amplitude_parameter"] = "H0"
    @test_throws ArgumentError MCMCConfig(stray_parameter)

    unknown_likelihood = example_dict()
    unknown_likelihood["likelihood"] = "marginalised"
    @test_throws ArgumentError MCMCConfig(unknown_likelihood)

    # The amplitude parameter is the reference value defining the template, so it must
    # have a fiducial.
    @test_throws ArgumentError MCMCConfig(marginalized_dict(;
        amplitude_parameter = "not_a_hyperparameter"))

    # Sampling and marginalizing the same parameter is silent double-counting.
    @test_throws ArgumentError MCMCConfig(marginalized_dict(;
        sample_only = ["H0", "Ωm"]))

    bad_nodes = marginalized_dict()
    bad_nodes["amplitude_num_nodes"] = 1
    @test_throws ArgumentError MCMCConfig(bad_nodes)

    bad_span = marginalized_dict()
    bad_span["amplitude_prior_span_sigma"] = 0.0
    @test_throws ArgumentError MCMCConfig(bad_span)
end

@testset "posterior_params" begin
    # Under the default likelihood the saved chain and the sampler's latents coincide.
    @test posterior_params(example_config(; sample_only = [:H0, :Ωm])) == [:H0, :Ωm]
    @test posterior_params(example_config(; sample_only = nothing)) == Symbol[]

    # Under marginalization they differ by exactly the amplitude parameter: no latent for
    # it, but it *is* written into the posterior group by the reconstruction.
    cfg = example_config(; sample_only = [:Ωm], likelihood = "amplitude_marginalized",
        amplitude_parameter = :H0)
    @test cfg.sample_only == [:Ωm]
    @test posterior_params(cfg) == [:Ωm, :H0]
    # It returns a copy: mutating the result must not corrupt the config.
    push!(posterior_params(cfg), :bogus)
    @test cfg.sample_only == [:Ωm]
end

@testset "NETCDF_PARAMETER_NAMES" begin
    # Every Unicode hyperparameter the runner's hyperprior declares needs an ASCII name,
    # or it lands in the file under a name the Python stack cannot match.
    for name in (:Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak, :R₀)
        @test haskey(NETCDF_PARAMETER_NAMES, name)
        @test isascii(String(NETCDF_PARAMETER_NAMES[name]))
    end
    # Already-ASCII names fall through unchanged rather than being listed redundantly.
    for name in (:H0, :w0, :total_merger_rate, :amplitude_mle, :template_optimal_snr)
        @test get(NETCDF_PARAMETER_NAMES, name, name) === name
    end
    @test allunique(values(NETCDF_PARAMETER_NAMES))
end
