module ASGWBInference

using Comonicon: @cast, @main

include("InferenceImpl.jl")
using .InferenceImpl:
                      ASGWBLogDensity,
                      unconstrained_initial_point,
                      constrained_parameters,
                      ad_logdensity,
                      finite_difference_logdensity_and_gradient,
                      sample_with_advancedhmc,
                      build_turing_model,
                      sample_with_turing,
                      condition_turing_model

export ASGWBLogDensity,
       unconstrained_initial_point,
       constrained_parameters,
       ad_logdensity,
       finite_difference_logdensity_and_gradient,
       sample_with_advancedhmc,
       build_turing_model,
       sample_with_turing,
       condition_turing_model

include("cli/run_inference.jl")
include("cli/stack_partial_chains.jl")
include("cli/profile_turing_main.jl")

@cast function mcmc(;
        config::String = "",
        seed::Int = 42,
        output_dir::String = "",
        output_prefix::String = "",
        num_chains::Int = -1,
        n_samples::Int = 0,
        n_adapts::Int = 0,
        checkpoint_every::Int = -1,
        interactive::Bool = false
)
    return RunInferenceCLI.run(;
        config,
        seed,
        output_dir,
        output_prefix,
        num_chains,
        n_samples,
        n_adapts,
        checkpoint_every,
        interactive
    )
end

@cast function stack_chains(
        inputs::String...;
        output::String,
        force::Bool = false
)
    return StackPartialChainsCLI.stack(inputs...; output, force)
end

@cast function profile(;
        config_file::String,
        seconds::Float64 = 2.0,
        profile_samples::Int = 500,
        alloc::Bool = false,
        profile_out::String = ""
)
    return ASGWBProfileMainCLI.profile(;
        config_file,
        seconds,
        profile_samples,
        alloc,
        profile_out
    )
end

@main

end
