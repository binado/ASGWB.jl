# Profile the AstroSGWB Turing log-density to find the bottleneck
# inside a NUTS gradient evaluation.
#
# Run from the repository root, for example:
#   julia --project=scripts/run scripts/profile_turing.jl --config-file=config/profile_turing.toml
#
# Optional: --seconds=2.0 --profile-samples=500 --alloc --profile-out=profile.dat
#
# The catalog is a mandatory real `catalog.h5`, loaded with `load_catalog` exactly
# as the production notebooks do. Test fixtures are deliberately *not* supported:
# their tiny sample/frequency counts (e.g. 2 samples, 3 bins) make the fixed-cost
# redshift-prior build dominate, which is wildly unrepresentative of production
# runs (~10⁴ samples, ~10² bins) and produces misleading bottleneck rankings.
#
# This script is *measurement only*: it does not edit any AstroSGWB/src/ files.

module AstroSGWBProfileCLI

using Distributions: logpdf, product_distribution, Uniform
using AstroSGWB
using AstroSGWBInference: build_turing_model, fiducial_spectral_density, logposterior
using AstroSGWBInference: merger_rate_and_log_weights
using AstroSGWBImportanceModels:
                                 bns_madau_dickinson_hyperparameters,
                                 bns_samples_from_catalog,
                                 prepare_bns_madau_dickinson_model
using AstroSGWB:
                 merger_rate_per_sec,
                 spectral_density,
                 MadauDickinsonSourceFrame,
                 CosmologyCache,
                 redshift,
                 canonical_hyperparameters,
                 cosmology,
                 luminosity_distance,
                 build_redshift_prior,
                 source_frame_distribution,
                 load_catalog,
                 frequencies,
                 in_band_mask,
                 build_observation_context,
                 ModifiedPropagation,
                 LambdaCDM,
                 Detector
using BenchmarkTools
using DelimitedFiles
using LogDensityProblems
using LogDensityProblemsAD
using Printf
using Profile
using Random
using Serialization
using Statistics: mean
using TOML
using Turing: DynamicPPL

# ---------------------------------------------------------------------------
# TOML config helpers
# ---------------------------------------------------------------------------

function _require(settings::Dict, key::AbstractString)
    haskey(settings, key) || throw(ArgumentError("missing required TOML key $(repr(key))"))
    return settings[key]
end

function _require_table(settings::Dict, key::AbstractString)
    v = _require(settings, key)
    v isa Dict || throw(ArgumentError("TOML key $(repr(key)) must be a table"))
    return v
end

function _require_string_array(settings::Dict, key::AbstractString)
    v = _require(settings, key)
    v isa Vector || throw(ArgumentError("TOML key $(repr(key)) must be an array"))
    all(x -> x isa AbstractString, v) ||
        throw(ArgumentError("TOML key $(repr(key)) must be an array of strings"))
    return Vector{String}(v)
end

function _resolve_catalog_path(catalog_path::String, base::AbstractString)
    resolved = isabspath(catalog_path) ? catalog_path :
               normpath(joinpath(base, catalog_path))
    isfile(resolved) || throw(
        ArgumentError(
        "catalog file not found: $(repr(resolved)). The profiler requires a real " *
        "catalog.h5 (loaded like the production notebooks); test fixtures are not " *
        "supported because their tiny sample counts give misleading bottlenecks.",
    ),
    )
    return resolved
end

function _load_observed_spectral_density(path::AbstractString, expected_len::Int)
    isfile(path) || throw(ArgumentError("observed spectrum file not found: $(repr(path))"))
    v = vec(readdlm(path, ',', Float64))
    length(v) == expected_len || throw(
        ArgumentError(
        "observed_spectral_density_csv has length $(length(v)), expected $expected_len",
    ),
    )
    return v
end

function _uniform_bounds(priors_tbl::Dict, key::AbstractString)
    sub = priors_tbl[key]
    sub isa Dict ||
        throw(ArgumentError("priors.$key must be a table with 'low' and 'high'"))
    lo = Float64(sub["low"])
    hi = Float64(sub["high"])
    isfinite(lo) && isfinite(hi) ||
        throw(ArgumentError("priors.$key: low and high must be finite"))
    lo < hi || throw(ArgumentError("priors.$key: require low < high, got ($lo, $hi)"))
    return lo, hi
end

function _priors_from_toml(priors_tbl::Dict)
    return product_distribution((
        H0 = Uniform(_uniform_bounds(priors_tbl, "H0")...),
        Ωm = Uniform(_uniform_bounds(priors_tbl, "Omega_m")...),
        Ξ₀ = Uniform(_uniform_bounds(priors_tbl, "Xi_0")...),
        Ξₙ = Uniform(_uniform_bounds(priors_tbl, "Xi_n")...),
        γ = Uniform(_uniform_bounds(priors_tbl, "gamma")...),
        κ = Uniform(_uniform_bounds(priors_tbl, "kappa")...),
        zpeak = Uniform(_uniform_bounds(priors_tbl, "z_peak")...)
    ))
end

function _theta0_from_toml(init_tbl::Dict, order::Tuple{Vararg{Symbol}})
    return canonical_hyperparameters(
        order,
        (;
            H0 = init_tbl["H0"],
            Ωm = init_tbl["Omega_m"],
            Ξ₀ = init_tbl["Xi_0"],
            Ξₙ = init_tbl["Xi_n"],
            γ = init_tbl["gamma"],
            κ = init_tbl["kappa"],
            zpeak = init_tbl["z_peak"]
        );
        context = "initial hyperparameters"
    )
end

function _validate_init_in_priors(prior, init_tbl::Dict)
    for (key, sym) in (
        "H0" => :H0,
        "Omega_m" => :Ωm,
        "Xi_0" => :Ξ₀,
        "Xi_n" => :Ξₙ,
        "gamma" => :γ,
        "kappa" => :κ,
        "z_peak" => :zpeak
    )
        haskey(init_tbl, key) || continue
        v = Float64(init_tbl[key])
        isfinite(logpdf(prior.dists[sym], v)) || throw(
            ArgumentError("init.$key = $v is outside the support of the corresponding prior"),
        )
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Format a nanosecond count as a human-readable duration."""
function _fmt_time(ns::Real)
    if ns < 1e3
        return @sprintf("%.1f ns", ns)
    elseif ns < 1e6
        return @sprintf("%.2f µs", ns / 1e3)
    elseif ns < 1e9
        return @sprintf("%.2f ms", ns / 1e6)
    else
        return @sprintf("%.3f s", ns / 1e9)
    end
end

function _fmt_bytes(b::Real)
    b < 1024 ? @sprintf("%d B", b) :
    b < 1024^2 ? @sprintf("%.1f KiB", b / 1024) :
    b < 1024^3 ? @sprintf("%.2f MiB", b / 1024^2) : @sprintf("%.2f GiB", b / 1024^3)
end

"""Median time (ns) from a BenchmarkTools.Trial."""
_median_ns(t::BenchmarkTools.Trial) = time(median(t))
_min_ns(t::BenchmarkTools.Trial) = time(minimum(t))
_mean_ns(t::BenchmarkTools.Trial) = time(mean(t))

function _print_trial_row(
        name::AbstractString,
        t::BenchmarkTools.Trial;
        pct_of::Union{Nothing, Real} = nothing
)
    med = _median_ns(t)
    pct_str = pct_of === nothing ? "" : @sprintf("  (%.1f%%)", 100 * med / pct_of)
    @info @sprintf("  %-28s  median=%s  min=%s  mean=%s  allocs=%d  mem=%s%s",
        name,
        _fmt_time(med),
        _fmt_time(_min_ns(t)),
        _fmt_time(_mean_ns(t)),
        allocs(median(t)),
        _fmt_bytes(memory(median(t))),
        pct_str,)
end

# ---------------------------------------------------------------------------
# Turing DPL log-density plumbing
# ---------------------------------------------------------------------------

"""
Build a DynamicPPL `LogDensityFunction` with a linked (unconstrained) `VarInfo`
seeded from the model's prior. Returns (lf, z0_turing) ready for
`LogDensityProblems.logdensity(lf, z0_turing)`.
"""
function _build_turing_logdensity(model)
    vi = DynamicPPL.VarInfo(model)
    vi_linked = DynamicPPL.link(vi, model)
    # `getlogjoint_internal` tells LogDensityFunction to return the joint
    # log-density in *internal* (unconstrained, post-link) parameterization,
    # which is what NUTS actually evaluates.
    lf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi_linked)
    z = convert(Vector{Float64}, vi_linked[:])
    return lf, z
end

# ---------------------------------------------------------------------------
# Main profiling entrypoint
# ---------------------------------------------------------------------------

function _run(;
        catalog_path::String,
        detectors::Vector{Detector},
        priors,
        init_tbl::Dict,
        seed::Union{Nothing, Int},
        observed_spectral_density_csv::Union{Nothing, String},
        local_merger_rate::Float64,
        observation_time::Float64,
        seconds::Float64,
        profile_samples::Int,
        do_alloc::Bool,
        profile_out::Union{Nothing, String}
)
    t0 = time()

    @info "loading catalog" catalog_path detectors=join((d.name for d in detectors), ",")
    loaded = load_catalog(catalog_path)
    C = LambdaCDM
    P = ModifiedPropagation
    order = bns_madau_dickinson_hyperparameters(C, P)
    θ0 = _theta0_from_toml(init_tbl, order)
    samples = bns_samples_from_catalog(loaded.catalog.samples, C, θ0)
    fluxes = loaded.catalog.fluxes
    model = prepare_bns_madau_dickinson_model(
        samples,
        θ0,
        C,
        P;
        observation_time = observation_time,
        local_merger_rate = local_merger_rate
    )
    observation = build_observation_context(
        frequencies(loaded.metadata.grid), detectors,
        in_band_mask(loaded.metadata.grid), observation_time)
    @info "catalog loaded" n_frequency_bins=length(observation.frequencies) n_proposal_samples=length(samples.redshift)

    observed = if observed_spectral_density_csv === nothing
        @info "using fiducial spectrum from catalog as observed data"
        fiducial_spectral_density(model, fluxes, samples, θ0)
    else
        @info "loading observed spectrum from CSV" path = observed_spectral_density_csv
        _load_observed_spectral_density(
            observed_spectral_density_csv,
            length(observation.frequencies)
        )
    end

    if seed !== nothing
        @info "seeding RNG" seed = seed
        Random.seed!(seed)
    end

    # ------------------------------------------------------------------
    # Build callables
    # ------------------------------------------------------------------

    # Turing / DynamicPPL path
    turing_model = build_turing_model(
        model,
        fluxes,
        samples,
        θ0,
        observation,
        priors;
        track = false,
        observed = observed
    )
    lf, z0_turing = _build_turing_logdensity(turing_model)
    ad_lf = LogDensityProblemsAD.ADgradient(:ForwardDiff, lf)

    # Intermediate values frozen at θ0 for stage-level benchmarks
    h = θ0
    c0 = cosmology(C, h)
    cache0 = CosmologyCache(c0, model.z_grid)
    sfn0 = zz -> source_frame_distribution(MadauDickinsonSourceFrame(), zz, h)
    redshift_prior0 = build_redshift_prior(sfn0, cache0)
    rate0, log_weights0 = merger_rate_and_log_weights(model, h, samples)
    weights0 = exp.(log_weights0)
    z_samples = redshift(samples)

    # ------------------------------------------------------------------
    # Warmup — trigger JIT / AD compilation before any benchmark
    # ------------------------------------------------------------------
    @info "warming up (JIT + AD compile)"
    LogDensityProblems.logdensity(lf, z0_turing)
    LogDensityProblems.logdensity_and_gradient(ad_lf, z0_turing)
    logposterior(h, model, fluxes, samples, observation, priors, observed)

    # ------------------------------------------------------------------
    # BenchmarkTools suite
    # ------------------------------------------------------------------
    # Note on `$`-interpolation inside @benchmarkable: it pins each argument
    # into the generated benchmark function so BenchmarkTools doesn't pay
    # for a global-variable lookup inside the inner measurement loop
    # (the single most common BenchmarkTools footgun).

    suite = BenchmarkGroup()

    suite["primal"] = BenchmarkGroup()
    suite["primal"]["turing"] = @benchmarkable LogDensityProblems.logdensity($lf, $z0_turing)
    suite["primal"]["logposterior"] = @benchmarkable logposterior(
        $h,
        $model,
        $fluxes,
        $samples,
        $observation,
        $priors,
        $observed
    )

    suite["gradient"] = BenchmarkGroup()
    # gcsample=true forces a GC before each sample so AD timings are not
    # polluted by GC pauses from previous samples (AD allocates a lot).
    suite["gradient"]["turing"] = @benchmarkable(LogDensityProblems.logdensity_and_gradient($ad_lf, $z0_turing),
        gcsample = true)

    suite["stage"] = BenchmarkGroup()
    suite["stage"]["redshift"] = @benchmarkable build_redshift_prior(
        $sfn0, $cache0)
    # The fused joint replaces the separate weight/rate atomics: it returns
    # (rate, log_weights) in one cosmology-specific pass.
    suite["stage"]["rate_and_log_weights"] = @benchmarkable merger_rate_and_log_weights(
        $model, $h, $samples)
    suite["stage"]["rate"] = @benchmarkable merger_rate_per_sec(
        $redshift_prior0,
        $(model.local_merger_rate),
        $(model.observation_time)
    )
    suite["stage"]["spectral"] = @benchmarkable spectral_density(
        $fluxes,
        $rate0;
        weights = $weights0
    )
    suite["stage"]["prior"] = @benchmarkable logpdf($priors, $h)
    # Bare luminosity_distance broadcast — isolates per-sample distance work in
    # Catalog reconstruction and importance weighting.
    suite["stage"]["lumdist"] = @benchmarkable luminosity_distance.($z_samples, $c0)

    @info "tuning benchmark suite (evals/sample calibration)"
    tune!(suite)

    @info "running benchmark suite" seconds_per_entry = seconds
    results = run(suite; seconds = seconds, verbose = false)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @info "=== primal ==="
    t_primal_turing = results["primal"]["turing"]
    t_primal_logpost = results["primal"]["logposterior"]
    _print_trial_row("turing (DynamicPPL)", t_primal_turing)
    _print_trial_row("logposterior (bare)", t_primal_logpost)

    @info "=== gradient ==="
    t_grad_turing = results["gradient"]["turing"]
    _print_trial_row("turing (ForwardDiff)", t_grad_turing)

    # AD cost multiplier via BenchmarkTools.ratio
    r_turing = ratio(median(t_grad_turing), median(t_primal_turing))
    @info @sprintf("AD multiplier (gradient/primal): turing=%.2fx", time(r_turing))

    @info "=== per-stage breakdown (denominator: median of logposterior primal) ==="
    primal_ns = _median_ns(t_primal_logpost)
    for key in ("redshift", "rate_and_log_weights", "rate", "spectral", "prior", "lumdist")
        _print_trial_row(key, results["stage"][key]; pct_of = primal_ns)
    end

    # ------------------------------------------------------------------
    # Sampling profile on the gradient
    # ------------------------------------------------------------------
    @info "sampling-profile: running $profile_samples Turing ForwardDiff gradient evals under Profile.@profile"
    Profile.clear()
    # 100µs sampling delay: one gradient eval is ~100µs, so the default 1ms
    # delay misses almost every sample. Pair with n=10^7 so we never run out
    # of buffer on longer profiling runs.
    Profile.init(; n = 10^7, delay = 1e-4)
    Profile.@profile begin
        for _ in 1:profile_samples
            LogDensityProblems.logdensity_and_gradient(ad_lf, z0_turing)
        end
    end

    @info "--- top 30 flat frames (by sample count) ---"
    Profile.print(
        IOContext(stdout, :color => false);
        format = :flat,
        sortedby = :count,
        mincount = 5,
        maxdepth = 20
    )

    @info "--- tree view (all threads; avoids idle-thread noise from groupby=:thread) ---"
    Profile.print(
        IOContext(stdout, :color => false);
        format = :tree,
        mincount = 5,
        maxdepth = 30
    )

    if profile_out !== nothing
        @info "writing raw profile snapshot" path = profile_out
        open(profile_out, "w") do io
            serialize(io, Profile.retrieve())
        end
    end

    # ------------------------------------------------------------------
    # Allocation profile
    # ------------------------------------------------------------------
    if do_alloc
        @info "allocation profile: running gradient evaluations under Profile.Allocs.@profile"
        Profile.Allocs.clear()
        # sample_rate=0.01 keeps overhead tractable while still catching
        # the dominant allocation sites.
        Profile.Allocs.@profile sample_rate = 0.01 begin
            for _ in 1:max(profile_samples, 50)
                LogDensityProblems.logdensity_and_gradient(ad_lf, z0_turing)
            end
        end
        allocs_snap = Profile.Allocs.fetch()
        total_bytes = sum(a.size for a in allocs_snap.allocs; init = 0)
        @info "allocation snapshot" nsamples=length(allocs_snap.allocs) total_bytes_sampled=_fmt_bytes(total_bytes)

        # Top N allocation types
        counts = Dict{Type, Tuple{Int, Int}}()
        for a in allocs_snap.allocs
            n, b = get(counts, a.type, (0, 0))
            counts[a.type] = (n + 1, b + a.size)
        end
        sorted = sort(collect(counts); by = x -> -x[2][2])
        @info "--- top 15 allocation types by sampled bytes ---"
        for (typ, (n, b)) in Iterators.take(sorted, 15)
            @info @sprintf("  %-50s  %8d allocs  %s", string(typ), n, _fmt_bytes(b))
        end
    end

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------
    println()
    println("## Profile summary")
    println()
    println("| section | stage | median | min | allocs | mem | %% of logposterior |")
    println("|---------|-------|--------|-----|--------|-----|-------------------|")
    _mdrow(section,
        stage,
        t;
        pct_of = nothing) = println(
        @sprintf("| %s | %s | %s | %s | %d | %s | %s |",
        section,
        stage,
        _fmt_time(_median_ns(t)),
        _fmt_time(_min_ns(t)),
        allocs(median(t)),
        _fmt_bytes(memory(median(t))),
        pct_of === nothing ? "-" : @sprintf("%.1f%%", 100 * _median_ns(t) / pct_of),)
    )
    _mdrow("primal", "turing", t_primal_turing; pct_of = primal_ns)
    _mdrow("primal", "logposterior", t_primal_logpost; pct_of = primal_ns)
    _mdrow("gradient", "turing", t_grad_turing; pct_of = primal_ns)
    for key in ("redshift", "rate_and_log_weights", "rate", "spectral", "prior", "lumdist")
        _mdrow("stage", key, results["stage"][key]; pct_of = primal_ns)
    end
    println()
    println(
        @sprintf("AD multiplier (gradient/primal): turing=%.2fx", time(r_turing))
    )
    println()
    println("Tip: save a baseline with")
    println("  BenchmarkTools.save(\"baseline.json\", results)")
    println("then compare later runs via")
    println(
        "  judge(median(run(suite)), median(BenchmarkTools.load(\"baseline.json\")[1]))",
    )

    @info "profile complete" total_seconds = round(time() - t0; digits = 2)
    return results
end

"""
Profile the AstroSGWB Turing log-density to localize the NUTS bottleneck.

Uses BenchmarkTools for timing and `Profile` (stdlib) for sampling/allocation profiles.

# Options

- `-c, --config-file=<path>`: TOML settings file.

- `--seconds=<float>`: wall-time budget per benchmark entry (default 2.0).

- `--profile-samples=<int>`: number of gradient evals under `Profile.@profile` (default 500).

- `--alloc`: also run an allocation profile via `Profile.Allocs`.

- `--profile-out=<path>`: write raw `Profile.retrieve()` snapshot via `Serialization`.
"""
function profile_turing(;
        config_file::String,
        seconds::Float64 = 2.0,
        profile_samples::Int = 500,
        alloc::Bool = false,
        profile_out::String = ""
)
    @info "loading config" path = config_file
    cfg = TOML.parsefile(config_file)
    settings_dir = dirname(abspath(config_file))

    raw_catalog = _require(cfg, "catalog_path")::String
    catalog_path = _resolve_catalog_path(raw_catalog, settings_dir)
    detectors = [Detector(n) for n in _require_string_array(cfg, "detectors")]
    seed = get(cfg, "seed", nothing)
    observed_csv = get(cfg, "observed_spectral_density_csv", nothing)
    if observed_csv !== nothing
        observed_csv = String(observed_csv)
    end
    local_merger_rate = Float64(_require(cfg, "local_merger_rate"))
    observation_time = Float64(_require(cfg, "observation_time"))

    priors_tbl = _require_table(cfg, "priors")
    init_tbl = _require_table(cfg, "init")
    priors = _priors_from_toml(priors_tbl)
    _validate_init_in_priors(priors, init_tbl)

    @info "effective settings" catalog_path detectors=join(
        (d.name for d in detectors), ",") seed=seed

    return _run(;
        catalog_path,
        detectors,
        priors,
        init_tbl,
        seed,
        observed_spectral_density_csv = observed_csv,
        local_merger_rate,
        observation_time,
        seconds,
        profile_samples,
        do_alloc = alloc,
        profile_out = isempty(profile_out) ? nothing : profile_out
    )
end

function _pop_value!(args::Vector{String}, i::Int, option::String)
    i < length(args) || throw(ArgumentError("$option requires a value"))
    return args[i + 1], i + 2
end

function _parse_args(args::Vector{String})
    config_file = ""
    seconds = 2.0
    profile_samples = 500
    alloc = false
    profile_out = ""

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-c" || arg == "--config-file"
            config_file, i = _pop_value!(args, i, arg)
        elseif startswith(arg, "--config-file=")
            config_file = arg[(lastindex("--config-file=") + 1):end]
            i += 1
        elseif arg == "--seconds"
            value, i = _pop_value!(args, i, arg)
            seconds = parse(Float64, value)
        elseif startswith(arg, "--seconds=")
            seconds = parse(Float64, arg[(lastindex("--seconds=") + 1):end])
            i += 1
        elseif arg == "--profile-samples"
            value, i = _pop_value!(args, i, arg)
            profile_samples = parse(Int, value)
        elseif startswith(arg, "--profile-samples=")
            profile_samples = parse(Int, arg[(lastindex("--profile-samples=") + 1):end])
            i += 1
        elseif arg == "--alloc"
            alloc = true
            i += 1
        elseif arg == "--profile-out"
            profile_out, i = _pop_value!(args, i, arg)
        elseif startswith(arg, "--profile-out=")
            profile_out = arg[(lastindex("--profile-out=") + 1):end]
            i += 1
        else
            throw(ArgumentError("unknown argument: $arg"))
        end
    end

    isempty(config_file) && throw(ArgumentError("missing required --config-file=<path>"))
    return (;
        config_file,
        seconds,
        profile_samples,
        alloc,
        profile_out
    )
end

function command_main(args::Vector{String} = ARGS)::Cint
    try
        profile_turing(; _parse_args(copy(args))...)
        return Cint(0)
    catch err
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        return Cint(1)
    end
end

end # module AstroSGWBProfileCLI

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(Base.invokelatest(AstroSGWBProfileCLI.command_main))
end
