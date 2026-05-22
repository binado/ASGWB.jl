module ASGWBInference

include("InferenceImpl.jl")
using .InferenceImpl:
                      ASGWBLogDensity,
                      unconstrained_initial_point,
                      constrained_parameters,
                      ad_logdensity,
                      finite_difference_logdensity_and_gradient,
                      sample_with_advancedhmc,
                      build_turing_model,
                      condition_turing_model

export ASGWBLogDensity,
       unconstrained_initial_point,
       constrained_parameters,
       ad_logdensity,
       finite_difference_logdensity_and_gradient,
       sample_with_advancedhmc,
       build_turing_model,
       condition_turing_model,
       run_inference,
       run_inference_from_env,
       julia_main

include("cli/run_inference.jl")
include("cli/stack_partial_chains.jl")
include("cli/profile_turing_main.jl")

"""
    run_inference(config_path)

Run ASGWB inference from the TOML configuration at `config_path`.
Relative paths inside the TOML are resolved against the TOML file's directory.
"""
function run_inference(config_path::AbstractString)
    return RunInferenceCLI.run(config_path)
end

"""
    run_inference_from_env()

Run ASGWB inference using `MCMC_CONFIG_FILEPATH`, or
`config/run_inference.toml` relative to the repository root when the
environment variable is unset.
"""
function run_inference_from_env()
    return RunInferenceCLI.run_from_env()
end

"""
    julia_main()::Cint

PackageCompiler entrypoint. This executable is configured only through TOML
and environment variables; command-line arguments are rejected.
"""
function julia_main()::Cint
    if !isempty(ARGS)
        println(
            stderr,
            "ASGWBInference does not accept command-line arguments. ",
            "Use MCMC_CONFIG_FILEPATH or config/run_inference.toml for configuration."
        )
        return Cint(2)
    end

    try
        run_inference_from_env()
        return Cint(0)
    catch err
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        return Cint(1)
    end
end

end
