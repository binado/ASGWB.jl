# Profile the ASGWB Turing/AdvancedHMC log-density to find the bottleneck
# inside a NUTS gradient evaluation, before deciding whether to refactor
# the cosmology `quadgk` path.
#
# Run from the package root, for example:
#   julia --project=ASGWBInference -e 'using ASGWBInference; ASGWBInference.ASGWBProfileMainCLI.profile(config_file="config/profile_turing.toml")'
#
# Sites under investigation:
#   - ASGWB/src/cosmology.jl:9-13     quadgk inside comoving_distance
#   - ASGWB/src/redshift.jl           differential_comoving_volume.(z_grid, H0, Ωm) broadcast
#   - ASGWB/src/importance.jl         luminosity_distance.(z, H0, Ωm) per-sample broadcast
#   - ASGWBInference/src/turing_model.jl:55-97 the model being profiled
#
# This script is *measurement only*: it does not edit any ASGWB/src/ files.

module ASGWBProfileMainCLI

const _REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const _PARITY_TEST_CACHE = joinpath(_REPO_ROOT, "ASGWB", "test", "parity_test_cache.jl")

using Distributions: logpdf, product_distribution, Uniform
using ASGWB
using ASGWB:
             load_cache,
             cosmology_and_redshift_prior,
             compute_importance_weights,
             merger_rate_per_sec,
             spectral_density,
             logposterior,
             luminosity_distance,
             redshift,
             canonical_hyperparameters,
             MadauDickinsonModifiedPropagation,
             Detector
using ..InferenceImpl:
                       ASGWBLogDensity,
                       ad_logdensity,
                       unconstrained_initial_point,
                       build_turing_model
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
        Ξ₀ = Uniform(_uniform_bounds(priors_tbl, "chi0")...),
        Ξₙ = Uniform(_uniform_bounds(priors_tbl, "chin")...),
        γ = Uniform(_uniform_bounds(priors_tbl, "gamma")...),
        κ = Uniform(_uniform_bounds(priors_tbl, "kappa")...),
        zpeak = Uniform(_uniform_bounds(priors_tbl, "z_peak")...)
    ))
end

function _theta0_from_toml(init_tbl::Dict)
    return canonical_hyperparameters(
        MadauDickinsonModifiedPropagation(),
        (;
            H0 = init_tbl["H0"],
            Ωm = init_tbl["Omega_m"],
            Ξ₀ = init_tbl["chi0"],
            Ξₙ = init_tbl["chin"],
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
        "chi0" => :Ξ₀,
        "chin" => :Ξₙ,
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
        cache_path::String,
        detectors::Vector{Detector},
        priors,
        θ0::NamedTuple,
        seed::Union{Nothing, Int},
        observed_spectral_density_csv::Union{Nothing, String},
        seconds::Float64,
        profile_samples::Int,
        do_alloc::Bool,
        profile_out::Union{Nothing, String}
)
    t0 = time()

    @info "loading importance cache" path=cache_path detectors=join(
        (d.name
        for d in detectors), ",")
    problem = load_cache(cache_path, detectors)
    @info "cache loaded" n_frequency_bins=length(problem.observation.frequencies) n_proposal_samples=length(problem.proposal.samples.redshift)

    observed = if observed_spectral_density_csv === nothing
        @info "using fiducial in-band spectrum from cache as observed data"
        problem.observation.fiducial_spectral_density
    else
        @info "loading observed spectrum from CSV" path = observed_spectral_density_csv
        _load_observed_spectral_density(
            observed_spectral_density_csv,
            length(problem.observation.fiducial_spectral_density)
        )
    end

    if seed !== nothing
        @info "seeding RNG" seed = seed
        Random.seed!(seed)
    end

    # ------------------------------------------------------------------
    # Build callables
    # ------------------------------------------------------------------

    # Native (ASGWBLogDensity) path — pure Julia, no DynamicPPL
    ld = ASGWBLogDensity(problem, priors)
    z0 = unconstrained_initial_point(ld, θ0)
    ad_ld = ad_logdensity(ld)

    # Turing / DynamicPPL path
    model = build_turing_model(problem, priors; track = false, observed_spectral_density = observed)
    lf, z0_turing = _build_turing_logdensity(model)
    ad_lf = LogDensityProblemsAD.ADgradient(:ForwardDiff, lf)

    # Intermediate values frozen at θ0 for stage-level benchmarks
    h = θ0
    spec = problem.redshift_prior_spec
    cosmology_cache0,
    redshift_prior0 = cosmology_and_redshift_prior(
        h, spec, problem.redshift_cache.redshift_grid)
    iw0 = compute_importance_weights(problem, h, cosmology_cache0, redshift_prior0)
    rate0 = merger_rate_per_sec(
        redshift_prior0,
        problem.local_merger_rate,
        problem.observation.observation_time_yr,
        problem.observation.observation_time_sec
    )
    weights0 = iw0.weights
    z_samples = redshift(problem)
    H0_val = h.H0
    Ωm_val = h.Ωm

    # ------------------------------------------------------------------
    # Warmup — trigger JIT / AD compilation before any benchmark
    # ------------------------------------------------------------------
    @info "warming up (JIT + AD compile)"
    LogDensityProblems.logdensity(ld, z0)
    LogDensityProblems.logdensity_and_gradient(ad_ld, z0)
    LogDensityProblems.logdensity(lf, z0_turing)
    LogDensityProblems.logdensity_and_gradient(ad_lf, z0_turing)
    logposterior(h, problem, priors; observed_spectral_density = observed)

    # ------------------------------------------------------------------
    # BenchmarkTools suite
    # ------------------------------------------------------------------
    # Note on `$`-interpolation inside @benchmarkable: it pins each argument
    # into the generated benchmark function so BenchmarkTools doesn't pay
    # for a global-variable lookup inside the inner measurement loop
    # (the single most common BenchmarkTools footgun).

    suite = BenchmarkGroup()

    suite["primal"] = BenchmarkGroup()
    suite["primal"]["native"] = @benchmarkable LogDensityProblems.logdensity($ld, $z0)
    suite["primal"]["turing"] = @benchmarkable LogDensityProblems.logdensity($lf, $z0_turing)
    suite["primal"]["logposterior"] = @benchmarkable logposterior(
        $h,
        $problem,
        $priors;
        observed_spectral_density = $observed
    )

    suite["gradient"] = BenchmarkGroup()
    # gcsample=true forces a GC before each sample so AD timings are not
    # polluted by GC pauses from previous samples (AD allocates a lot).
    suite["gradient"]["native"] = @benchmarkable(LogDensityProblems.logdensity_and_gradient($ad_ld, $z0),
        gcsample = true)
    suite["gradient"]["turing"] = @benchmarkable(LogDensityProblems.logdensity_and_gradient($ad_lf, $z0_turing),
        gcsample = true)

    suite["stage"] = BenchmarkGroup()
    suite["stage"]["redshift"] = @benchmarkable cosmology_and_redshift_prior(
        $h, $spec, $(problem.redshift_cache.redshift_grid))
    suite["stage"]["weights"] = @benchmarkable compute_importance_weights(
        $problem, $h, $cosmology_cache0, $redshift_prior0)
    suite["stage"]["rate"] = @benchmarkable merger_rate_per_sec(
        $redshift_prior0,
        $(problem.local_merger_rate),
        $(problem.observation.observation_time_yr),
        $(problem.observation.observation_time_sec)
    )
    suite["stage"]["spectral"] = @benchmarkable spectral_density(
        $(problem.proposal.cached_flux_over_dgw2),
        $rate0;
        weights = $weights0
    )
    suite["stage"]["prior"] = @benchmarkable logpdf($priors, $h)
    # Bare luminosity_distance broadcast — isolates the per-sample quadgk
    # cost currently in ASGWB/src/importance.jl:36 and ASGWB/src/cache.jl:70.
    suite["stage"]["lumdist"] = @benchmarkable luminosity_distance.($z_samples, $H0_val, $Ωm_val)

    @info "tuning benchmark suite (evals/sample calibration)"
    tune!(suite)

    @info "running benchmark suite" seconds_per_entry = seconds
    results = run(suite; seconds = seconds, verbose = false)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @info "=== primal ==="
    t_primal_native = results["primal"]["native"]
    t_primal_turing = results["primal"]["turing"]
    t_primal_logpost = results["primal"]["logposterior"]
    _print_trial_row("native (ASGWBLogDensity)", t_primal_native)
    _print_trial_row("turing (DynamicPPL)", t_primal_turing)
    _print_trial_row("logposterior (bare)", t_primal_logpost)

    @info "=== gradient ==="
    t_grad_native = results["gradient"]["native"]
    t_grad_turing = results["gradient"]["turing"]
    _print_trial_row("native (ForwardDiff)", t_grad_native)
    _print_trial_row("turing (ForwardDiff)", t_grad_turing)

    # AD cost multiplier via BenchmarkTools.ratio
    r_native = ratio(median(t_grad_native), median(t_primal_native))
    r_turing = ratio(median(t_grad_turing), median(t_primal_turing))
    @info @sprintf("AD multiplier (gradient/primal): native=%.2fx  turing=%.2fx",
        time(r_native),
        time(r_turing),)

    @info "=== per-stage breakdown (denominator: median of logposterior primal) ==="
    primal_ns = _median_ns(t_primal_logpost)
    for key in ("redshift", "weights", "rate", "spectral", "prior", "lumdist")
        _print_trial_row(key, results["stage"][key]; pct_of = primal_ns)
    end

    # ------------------------------------------------------------------
    # Sampling profile on the gradient
    # ------------------------------------------------------------------
    # NUTS evaluates the same log-posterior body as `logposterior`; the native
    # `ASGWBLogDensity` + ForwardDiff path is used here so hot spots map cleanly
    # to `src/` without DynamicPPL stack noise (benchmarks above already compare
    # native vs Turing gradient wall times).
    @info "sampling-profile: running $profile_samples native ForwardDiff gradient evals under Profile.@profile"
    Profile.clear()
    # 100µs sampling delay: one gradient eval is ~100µs, so the default 1ms
    # delay misses almost every sample. Pair with n=10^7 so we never run out
    # of buffer on longer profiling runs.
    Profile.init(; n = 10^7, delay = 1e-4)
    Profile.@profile begin
        for _ in 1:profile_samples
            LogDensityProblems.logdensity_and_gradient(ad_ld, z0)
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
                LogDensityProblems.logdensity_and_gradient(ad_ld, z0)
            end
        end
        allocs_snap = Profile.Allocs.fetch()
        total_bytes = sum(a.size for a in allocs_snap.allocs; init = 0)
        @info "allocation snapshot" n_samples=length(allocs_snap.allocs) total_bytes_sampled=_fmt_bytes(total_bytes)

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
    _mdrow("primal", "native", t_primal_native; pct_of = primal_ns)
    _mdrow("primal", "turing", t_primal_turing; pct_of = primal_ns)
    _mdrow("primal", "logposterior", t_primal_logpost; pct_of = primal_ns)
    _mdrow("gradient", "native", t_grad_native; pct_of = primal_ns)
    _mdrow("gradient", "turing", t_grad_turing; pct_of = primal_ns)
    for key in ("redshift", "weights", "rate", "spectral", "prior", "lumdist")
        _mdrow("stage", key, results["stage"][key]; pct_of = primal_ns)
    end
    println()
    println(
        @sprintf("AD multiplier (gradient/primal): native=%.2fx, turing=%.2fx",
        time(r_native),
        time(r_turing))
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
Profile the ASGWB Turing/AdvancedHMC log-density to localize the NUTS bottleneck.

Uses BenchmarkTools for timing and `Profile` (stdlib) for sampling/allocation profiles.

# Options

- `-c, --config-file=<path>`: TOML settings file.

- `--seconds=<float>`: wall-time budget per benchmark entry (default 2.0).

- `--profile-samples=<int>`: number of gradient evals under `Profile.@profile` (default 500).

- `--alloc`: also run an allocation profile via `Profile.Allocs`.

- `--profile-out=<path>`: write raw `Profile.retrieve()` snapshot via `Serialization`.
"""
function profile(;
        config_file::String,
        seconds::Float64 = 2.0,
        profile_samples::Int = 500,
        alloc::Bool = false,
        profile_out::String = ""
)
    @info "loading config" path = config_file
    cfg = TOML.parsefile(config_file)

    raw_cache = _require(cfg, "cache_path")::String
    cache_path = if startswith(raw_cache, "parity:")
        isdefined(@__MODULE__, :resolve_parity_cache_path) ||
            include(_PARITY_TEST_CACHE)
        resolve_parity_cache_path(raw_cache)
    else
        raw_cache
    end
    detectors = [Detector(n) for n in _require_string_array(cfg, "detectors")]
    seed = get(cfg, "seed", nothing)
    observed_csv = get(cfg, "observed_spectral_density_csv", nothing)
    if observed_csv !== nothing
        observed_csv = String(observed_csv)
    end

    priors_tbl = _require_table(cfg, "priors")
    init_tbl = _require_table(cfg, "init")
    priors = _priors_from_toml(priors_tbl)
    _validate_init_in_priors(priors, init_tbl)
    θ0 = _theta0_from_toml(init_tbl)

    @info "effective settings" cache=cache_path detectors=join((d.name for d in detectors), ",") seed=seed

    return _run(;
        cache_path,
        detectors,
        priors,
        θ0,
        seed,
        observed_spectral_density_csv = observed_csv,
        seconds,
        profile_samples,
        do_alloc = alloc,
        profile_out = isempty(profile_out) ? nothing : profile_out
    )
end

end # module ASGWBProfileMainCLI
