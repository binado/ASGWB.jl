using Test
using TOML
using ASGWB
using ASGWBInference.InferenceImpl: POPULATION_REGISTRY, validate_hyperprior,
                                    load_problem_context

@testset "profile_turing script loads with current PopulationModel API" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    include(joinpath(repo, "scripts", "profile_turing.jl"))

    cfg_path = joinpath(repo, "config", "profile_turing.toml")
    cfg = TOML.parsefile(cfg_path)
    settings_dir = dirname(abspath(cfg_path))

    bundle_path,
    model_path = ASGWBProfileCLI._resolve_problem_paths(
        cfg["bundle_path"], cfg["model_path"], settings_dir)
    detectors = [Detector(n)
                 for n in ASGWBProfileCLI._require_string_array(cfg, "detectors")]
    loaded = load_problem_context(
        bundle_path,
        model_path,
        detectors,
        POPULATION_REGISTRY;
        local_merger_rate = Float64(cfg["local_merger_rate"]),
        observation_time_yr = Float64(cfg["observation_time_yr"])
    )
    problem = loaded.problem

    order = full_hyperparameters(loaded.cosmology_type, problem.population_model)
    priors = ASGWBProfileCLI._priors_from_toml(ASGWBProfileCLI._require_table(cfg, "priors"))
    init_tbl = ASGWBProfileCLI._require_table(cfg, "init")
    ASGWBProfileCLI._validate_init_in_priors(priors, init_tbl)
    θ0 = ASGWBProfileCLI._theta0_from_toml(init_tbl, order)

    @test validate_hyperprior(order, priors) === nothing
    @test order == (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    @test keys(θ0) == order
end
