# Run from the package root, for example:
#   julia --project=. scripts/run_turing.jl --config-file=scripts/examples/minimal_turing.json
#
# The CLI lives in a submodule so Comonicon does not call `command_main()` at parse time
# in `Main` (avoids Julia world-age issues when heavy packages load before the entry).

module ASGWBTuringCLI

using ASGWB
using ASGWB: build_uniform_priors, load_cache, sample_with_turing
using Comonicon: @main
using Random
using Serialization

include(joinpath(@__DIR__, "turing_settings.jl"))

function _run(s::Settings)
    validate_init_in_priors(s)
    problem = load_cache(s.cache)
    priors = build_uniform_priors(prior_dict(s))
    θ0 = theta0(s)
    observed = if s.observed_spectral_density_csv === nothing
        problem.observation.fiducial_spectral_density
    else
        load_observed_spectral_density(
            s.observed_spectral_density_csv,
            length(problem.observation.fiducial_spectral_density),
        )
    end
    if s.seed !== nothing
        Random.seed!(s.seed)
    end
    sam = s.sampler
    chain, _model = sample_with_turing(
        problem,
        priors,
        θ0;
        n_adapts = sam.n_adapts,
        n_samples = sam.n_samples,
        target_acceptance = sam.target_acceptance,
        observed_spectral_density = observed,
    )
    if s.output_jls !== nothing
        open(s.output_jls, "w") do io
            serialize(io, chain)
        end
        println("Wrote chain to ", s.output_jls)
    end
    println(
        "Done: chain size (iterations, params, chains) = ",
        size(chain),
    )
    return chain
end

"""
Run NUTS sampling for the ASGWB Turing importance model.

# Options

- `-c, --config-file=<path>`: JSON settings (cache path, priors, init, sampler, optional paths).

- `--cache=<path>`: override `cache` from the JSON file (empty string keeps JSON value).

- `--n-samples=<int>`: override `sampler.n_samples` (use a negative value, e.g. `-1`, to keep JSON).

- `--n-adapts=<int>`: override `sampler.n_adapts` (negative keeps JSON).

- `--target-acceptance=<float>`: override `sampler.target_acceptance` (negative keeps JSON).

- `--seed=<int>`: override RNG seed (negative keeps JSON / unset).

- `--observed-spectral-density-csv=<path>`: override observed spectrum CSV (empty keeps JSON).

- `--output-jls=<path>`: override output `.jls` path (empty keeps JSON).
"""
@main function run_turing(;
    config_file::String,
    cache::String = "",
    n_samples::Int = -1,
    n_adapts::Int = -1,
    target_acceptance::Float64 = -1.0,
    seed::Int = -1,
    observed_spectral_density_csv::String = "",
    output_jls::String = "",
)
    base = load_settings(config_file)
    s = merge_settings(
        base;
        cache = isempty(cache) ? nothing : cache,
        n_samples = n_samples < 0 ? nothing : n_samples,
        n_adapts = n_adapts < 0 ? nothing : n_adapts,
        target_acceptance = target_acceptance < 0 ? nothing : target_acceptance,
        seed = seed < 0 ? nothing : seed,
        observed_spectral_density_csv = isempty(observed_spectral_density_csv) ? nothing :
        observed_spectral_density_csv,
        output_jls = isempty(output_jls) ? nothing : output_jls,
    )
    return _run(s)
end

end # module ASGWBTuringCLI

Base.invokelatest(ASGWBTuringCLI.command_main)
