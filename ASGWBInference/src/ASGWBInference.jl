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

"""
    mcmc(; kwargs...)

Run ASGWB inference from a TOML configuration file.

# Options

- `--config=<path>`: TOML settings file. Falls back to `MCMC_CONFIG_FILEPATH`,
  then `ASGWBInference/run_inference.toml` next to the package directory.
- `--seed=<int>`: RNG seed (default: 42).
- `--output-dir=<path>`: Override `output_dir` from the TOML settings.
- `--output-prefix=<name>`: Override `output_prefix` from the TOML settings.
- `--num-chains=<int>`: Override `sampler.num_chains` from the TOML settings.
- `--n-samples=<int>`: Override `sampler.n_samples` from the TOML settings.
- `--n-adapts=<int>`: Override `sampler.n_adapts` from the TOML settings.
- `--checkpoint-every=<int>`: Override `sampler.checkpoint_every` from the TOML settings.
- `--interactive`: Enable Turing's sampling progress bar.

# Example

```bash
julia --project=ASGWBInference -m ASGWBInference mcmc --config=ASGWBInference/run_inference.toml
```
"""
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

"""
    stack_chains(inputs...; output, force=false)

Stack JLD2-saved MCMCChains files, such as per-chain checkpoint partials, into
one combined MCMCChains.Chains object.

# Args

- `inputs`: Input chain files or quoted glob patterns, for example
  `"chains.partial.chain*.jld2"`.

# Options

- `-o, --output=<path>`: Path for the stacked `.jld2` output.
- `-f, --force`: Overwrite an existing output file.

# Example

```bash
julia --project=ASGWBInference -m ASGWBInference stack-chains "partials*.jld2" --output=stacked.jld2
```
"""
@cast function stack_chains(
        inputs::String...;
        output::String,
        force::Bool = false
)
    return StackPartialChainsCLI.stack(inputs...; output, force)
end

"""
    profile(; kwargs...)

Profile the ASGWB Turing/AdvancedHMC log-density to localize the NUTS bottleneck.
Uses BenchmarkTools for timing and `Profile` (stdlib) for sampling/allocation profiles.

# Options

- `-c, --config-file=<path>`: TOML settings file.
- `--seconds=<float>`: Wall-time budget per benchmark entry (default 2.0).
- `--profile-samples=<int>`: Number of gradient evals under `Profile.@profile` (default 500).
- `--alloc`: Also run an allocation profile via `Profile.Allocs`.
- `--profile-out=<path>`: Write raw `Profile.retrieve()` snapshot via `Serialization`.

# Example

```bash
julia --project=ASGWBInference -m ASGWBInference profile --config-file=ASGWBInference/profile_turing.toml
```
"""
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
