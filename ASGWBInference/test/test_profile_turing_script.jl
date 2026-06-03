using Test
using TOML
using ASGWB
using ASGWBInference: validate_hyperprior

@testset "profile_turing script loads with current PopulationModel API" begin
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    include(joinpath(repo, "scripts", "profile_turing.jl"))

    cfg_path = joinpath(repo, "config", "profile_turing.toml")
    cfg = TOML.parsefile(cfg_path)
    settings_dir = dirname(abspath(cfg_path))

    catalog_path = ASGWBProfileCLI._resolve_catalog_path(cfg["catalog_path"], settings_dir)
    detectors = [Detector(n)
                 for n in ASGWBProfileCLI._require_string_array(cfg, "detectors")]
    loaded = load_catalog(catalog_path)
    pop = ASGWBProfileCLI.BNSPopulationModel()
    C = ModifiedPropagation{LambdaCDM}
    order = full_hyperparameters(C, pop)
    priors = ASGWBProfileCLI._priors_from_toml(ASGWBProfileCLI._require_table(cfg, "priors"))
    init_tbl = ASGWBProfileCLI._require_table(cfg, "init")
    ASGWBProfileCLI._validate_init_in_priors(priors, init_tbl)
    θ0 = ASGWBProfileCLI._theta0_from_toml(init_tbl, order)
    samples = ASGWBProfileCLI.bns_samples_from_catalog(loaded.catalog.samples)
    problem = ImportanceSamplingProblem(pop, loaded.catalog.fluxes, samples, θ0)
    ctx = build_model_context(
        problem,
        C,
        loaded.metadata.grid,
        detectors,
        Float64(cfg["observation_time_yr"]),
        Float64(cfg["local_merger_rate"])
    )

    @test validate_hyperprior(order, priors) === nothing
    @test order == (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    @test keys(θ0) == order
    @test length(ctx.fiducial_spectral_density) == length(ctx.observation.frequencies)
end
