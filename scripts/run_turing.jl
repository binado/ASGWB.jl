# Run from the package root, for example:
#   julia --project=. scripts/run_turing.jl --config-file=scripts/examples/minimal_turing.json
#
# H0-only smoke on the NumPyro-style cache (uses JSON `sample_only` and Turing `|` conditioning):
#   julia --project=. scripts/run_turing.jl --config-file=scripts/examples/numpyro_light_h0_turing.json
#
# The CLI lives in a submodule so Comonicon does not call `command_main()` at parse time
# in `Main` (avoids Julia world-age issues when heavy packages load before the entry).

module ASGWBTuringCLI

using ASGWB
using ASGWB: build_uniform_priors, load_cache, sample_with_turing
using Comonicon: @main
using Logging
using Random
using Serialization

include(joinpath(@__DIR__, "turing_settings.jl"))

function _run(s::Settings)
    t0 = time()
    @info "validating initial point against prior bounds"
    validate_init_in_priors(s)

    @info "loading importance cache" path=s.cache detectors=join(
        (d.name
        for d in s.detectors), ",")
    t_cache = time()
    problem = load_cache(s.cache, s.detectors)
    @info "cache loaded" seconds=round(time()-t_cache; digits = 2) n_frequency_bins=length(problem.observation.frequencies) n_proposal_samples=length(problem.proposal.samples.redshift)

    @info "building uniform priors and initial HyperParameters"
    priors = build_uniform_priors(prior_dict(s))
    θ0 = theta0(s)

    observed = if s.observed_spectral_density_csv === nothing
        @info "using fiducial in-band spectrum from cache as observed data"
        problem.observation.fiducial_spectral_density
    else
        @info "loading observed spectrum from CSV" path = s.observed_spectral_density_csv
        load_observed_spectral_density(
            s.observed_spectral_density_csv,
            length(problem.observation.fiducial_spectral_density)
        )
    end

    if s.seed !== nothing
        @info "seeding RNG" seed = s.seed
        Random.seed!(s.seed)
    else
        @info "RNG seed not set (nondeterministic run unless Julia was seeded elsewhere)"
    end

    sam = s.sampler
    sample_only_tup = if s.sample_only === nothing
        nothing
    else
        Tuple(s.sample_only)
    end
    @info "starting NUTS" n_adapts=sam.n_adapts n_samples=sam.n_samples target_acceptance=sam.target_acceptance sample_only=sample_only_tup

    t_sample = time()
    chain,
    _model = sample_with_turing(
        problem,
        priors,
        θ0;
        n_adapts = sam.n_adapts,
        n_samples = sam.n_samples,
        target_acceptance = sam.target_acceptance,
        observed_spectral_density = observed,
        sample_only = sample_only_tup
    )
    @info "NUTS finished" seconds=round(time()-t_sample; digits = 2) chain_size=size(chain)

    if s.output_jls !== nothing
        @info "serializing chain" path = s.output_jls
        open(s.output_jls, "w") do io
            serialize(io, chain)
        end
        @info "wrote chain to disk" path = s.output_jls
    end

    @info "run complete" total_seconds=round(time()-t0; digits = 2) chain_size=size(chain)
    return chain
end

"""
Run NUTS sampling for the ASGWB Turing importance model.

Progress is logged with `Logging.@info` (cache load time, sampler settings, NUTS wall time,
total runtime). Turing may still print its own short diagnostics during NUTS.

# Options

- `-c, --config-file=<path>`: JSON settings (cache path, priors, init, sampler, optional paths).

- `--cache=<path>`: override `cache` from the JSON file (empty string keeps JSON value).

- `--n-samples=<int>`: override `sampler.n_samples` (use a negative value, e.g. `-1`, to keep JSON).

- `--n-adapts=<int>`: override `sampler.n_adapts` (negative keeps JSON).

- `--target-acceptance=<float>`: override `sampler.target_acceptance` (negative keeps JSON).

- `--seed=<int>`: override RNG seed (negative keeps JSON / unset).

- `--observed-spectral-density-csv=<path>`: override observed spectrum CSV (empty keeps JSON).

- `--output-jls=<path>`: override output `.jls` path (empty keeps JSON).

- `--sample-only=<names>`: comma-separated hyperparameters to sample (e.g. `H0`); fixes the rest
  to `init` via Turing conditioning. Empty keeps JSON `sample_only` / full sampling.
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
        sample_only::String = ""
)
    @info "loading config" path = config_file
    base = load_settings(config_file)
    so_cli = parse_sample_only_cli(sample_only)
    s = merge_settings(
        base;
        cache = isempty(cache) ? nothing : cache,
        n_samples = n_samples < 0 ? nothing : n_samples,
        n_adapts = n_adapts < 0 ? nothing : n_adapts,
        target_acceptance = target_acceptance < 0 ? nothing : target_acceptance,
        seed = seed < 0 ? nothing : seed,
        observed_spectral_density_csv = isempty(observed_spectral_density_csv) ?
                                        nothing : observed_spectral_density_csv,
        output_jls = isempty(output_jls) ? nothing : output_jls,
        sample_only = so_cli
    )
    @info "effective settings" cache=s.cache detectors=join((d.name for d in s.detectors), ",") sample_only=s.sample_only n_adapts=s.sampler.n_adapts n_samples=s.sampler.n_samples target_acceptance=s.sampler.target_acceptance seed=s.seed output_jls=s.output_jls observed_spectral_density_csv=s.observed_spectral_density_csv
    return _run(s)
end

end # module ASGWBTuringCLI

Base.invokelatest(ASGWBTuringCLI.command_main)
