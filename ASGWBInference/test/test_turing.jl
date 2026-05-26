using Test
using Turing
using Turing.DynamicPPL: VarInfo, getsym
using FlexiChains
using ASGWB
using ASGWB: AbstractCosmology
using ASGWBInference: build_turing_model, condition_turing_model
using Distributions: product_distribution, Uniform

_varinfo_symbols(vi) = Set(getsym(vn) for vn in keys(vi))

if !@isdefined parity_cache_path
    include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_test_cache.jl"))
end
include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_fixtures.jl"))

@testset "Turing model smoke test" begin
    for variant in (:posterior, :full_intrinsic)
        cache = load_cache(parity_cache_path(variant), [Detector("H1"), Detector("L1")])
        theta0 = PARITY_THETA
        priors = PARITY_PRIORS
        prior_bounds = PARITY_PRIOR_BOUNDS

        model = build_turing_model(cache, priors; model = PARITY_MODEL, track = false)
        @test Turing.logjoint(model, theta0) ≈
              logposterior(theta0, cache, priors; model = PARITY_MODEL) rtol = 1e-6
        @test condition_turing_model(
            model, theta0, priors, nothing; model = PARITY_MODEL) ===
              model
        @test_throws ArgumentError condition_turing_model(
            model, theta0, priors, (); model = PARITY_MODEL)
        @test_throws ArgumentError condition_turing_model(
            model, theta0, priors, (:unknown,); model = PARITY_MODEL)

        model_track = build_turing_model(cache, priors; model = PARITY_MODEL, track = true)
        returned_nt = Turing.returned(model_track, theta0)
        @test haskey(returned_nt, :effective_sample_size)
        @test isfinite(returned_nt.effective_sample_size)
        @test 0 < returned_nt.effective_sample_size <= 1

        @test haskey(returned_nt, :spectral_snr_squared)
        @test haskey(returned_nt, :spectral_snr)
        @test isfinite(returned_nt.spectral_snr_squared)
        @test isfinite(returned_nt.spectral_snr)
        @test returned_nt.spectral_snr >= 0
        @test returned_nt.spectral_snr^2 ≈ returned_nt.spectral_snr_squared

        # Verify manual NUTS sampling flow
        sampled_model = condition_turing_model(
            model, theta0, priors, nothing; model = PARITY_MODEL)
        chain = sample(
            sampled_model,
            Turing.NUTS(3, 0.8),
            3;
            progress = false,
            chain_type = FlexiChains.VNChain
        )

        @test Turing.logjoint(sampled_model, theta0) ≈ Turing.logjoint(model, theta0) rtol = 1e-6
        @test chain isa FlexiChains.VNChain
        @test size(chain, 1) == 3
        @test sort(collect(Symbol.(FlexiChains.parameters(chain)))) ==
              sort(collect(keys(theta0)))

        cond_h0 = condition_turing_model(
            model, theta0, priors, (:H0,); model = PARITY_MODEL)
        chain_h0 = sample(
            cond_h0,
            Turing.NUTS(3, 0.8),
            3;
            progress = false,
            chain_type = FlexiChains.VNChain
        )
        @test chain_h0 isa FlexiChains.VNChain
        pnames = sort(collect(Symbol.(FlexiChains.parameters(chain_h0))))
        @test pnames == [:H0]
        @test all(isfinite, vec(Array(chain_h0[:logjoint])))

        _chain_sym = Dict(
            "H0" => :H0,
            "Omega_m" => :Ωm,
            "chi0" => :Ξ₀,
            "chin" => :Ξₙ,
            "gamma" => :γ,
            "kappa" => :κ,
            "z_peak" => :zpeak
        )
        for (name, (low, high)) in prior_bounds
            values = vec(Array(chain[_chain_sym[name]]))
            @test all(isfinite, values)
            @test all((low .<= values) .& (values .<= high))
        end

        @test all(isfinite, vec(Array(chain[:logjoint])))
    end
end

const _COSMO_PRIORS = (
    H0 = Uniform(20.0, 140.0),
    Ωm = Uniform(0.0, 1.0),
    w0 = Uniform(-3.0, 0.0),
    wa = Uniform(-3.0, 3.0),
    Ξ₀ = Uniform(0.0, 2.0),
    Ξₙ = Uniform(-1.0, 1.0),
    γ = Uniform(0.0, 5.0),
    κ = Uniform(0.0, 10.0),
    zpeak = Uniform(0.0, 5.0)
)

function _prior_for(model)
    names = hyperparameters(model)
    return product_distribution(NamedTuple{names}(_COSMO_PRIORS))
end

@testset "submodel boundary lifts VarNames to parent (flat)" begin
    cache = load_cache(parity_cache_path(:posterior), [Detector("H1"), Detector("L1")])

    cases = (
        (MadauDickinsonModifiedPropagation{LambdaCDM}(),
            (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)),
        (MadauDickinsonModifiedPropagation{W0CDM}(),
            (:H0, :Ωm, :w0, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)),
        (MadauDickinsonModifiedPropagation{W0WaCDM}(),
            (:H0, :Ωm, :w0, :wa, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak))
    )

    for (m, expected_names) in cases
        priors = _prior_for(m)
        turing_model = build_turing_model(cache, priors; model = m)
        vi = VarInfo(turing_model)
        present = _varinfo_symbols(vi)
        for n in expected_names
            @test n in present
        end
        # The submodel-returned NamedTuples must not surface as separate VarNames.
        @test !(:cosmo_nt in present)
        @test !(:model_nt in present)
    end
end

@testset "condition_turing_model across submodel boundary" begin
    cache = load_cache(parity_cache_path(:posterior), [Detector("H1"), Detector("L1")])
    m = MadauDickinsonModifiedPropagation{W0CDM}()
    priors = _prior_for(m)
    theta0 = canonical_hyperparameters(
        m,
        (; H0 = 70.0, Ωm = 0.3, w0 = -1.0, Ξ₀ = 1.1,
            Ξₙ = 0.2, γ = 2.9, κ = 6.0, zpeak = 2.2)
    )

    turing_model = build_turing_model(cache, priors; model = m)
    @test condition_turing_model(turing_model, theta0, priors, nothing; model = m) ===
          turing_model

    # Fix everything except w0; only w0 should remain stochastic.
    cond_w0 = condition_turing_model(turing_model, theta0, priors, (:w0,); model = m)
    sampled = _varinfo_symbols(VarInfo(cond_w0))
    @test :w0 in sampled
    for n in (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
        @test !(n in sampled)
    end

    # Fix everything except H0 (lifted from the cosmology submodel); only H0 stochastic.
    cond_H0 = condition_turing_model(turing_model, theta0, priors, (:H0,); model = m)
    sampled_H0 = _varinfo_symbols(VarInfo(cond_H0))
    @test :H0 in sampled_H0
    for n in (:Ωm, :w0, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
        @test !(n in sampled_H0)
    end
end
