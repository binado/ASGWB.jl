using Test
using Turing
using Turing.DynamicPPL: VarInfo, getsym
using FlexiChains
using ASGWB
using ASGWBInference: build_turing_model, condition_turing_model, logposterior,
                      POPULATION_REGISTRY
using Distributions: product_distribution, Uniform

_varinfo_symbols(vi) = Set(getsym(vn) for vn in keys(vi))

if !@isdefined parity_catalog_dir
    include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_test_cache.jl"))
end
include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_fixtures.jl"))

@testset "Turing model smoke test" begin
    for variant in (:posterior, :full_intrinsic)
        loaded = parity_problem_context(
            variant, [Detector("H1"), Detector("L1")]; registry = POPULATION_REGISTRY)
        cache, C, ctx = loaded.problem, loaded.cosmology_type, loaded.ctx
        theta0 = PARITY_THETA
        priors = PARITY_PRIORS
        order = _PARITY_ORDER

        model = build_turing_model(cache, C, ctx, priors; track = false)
        @test Turing.logjoint(model, theta0) ≈
              logposterior(theta0, cache, C, ctx, priors) rtol = 1e-6
        @test condition_turing_model(
            model, theta0, priors, nothing; order = order) ===
              model
        @test_throws ArgumentError condition_turing_model(
            model, theta0, priors, (); order = order)
        @test_throws ArgumentError condition_turing_model(
            model, theta0, priors, (:unknown,); order = order)

        model_track = build_turing_model(cache, C, ctx, priors; track = true)
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

        sampled_model = condition_turing_model(
            model, theta0, priors, nothing; order = order)
        chain = sample(
            sampled_model,
            Turing.NUTS(3, 0.8),
            3;
            progress = false,
            chain_type = FlexiChains.VNChain,
            initial_params = InitFromPrior()
        )

        @test Turing.logjoint(sampled_model, theta0) ≈ Turing.logjoint(model, theta0) rtol = 1e-6
        @test chain isa FlexiChains.VNChain
        @test size(chain, 1) == 3
        @test sort(collect(Symbol.(FlexiChains.parameters(chain)))) ==
              sort(collect(keys(theta0)))

        cond_h0 = condition_turing_model(
            model, theta0, priors, (:H0,); order = order)
        chain_h0 = sample(
            cond_h0,
            Turing.NUTS(3, 0.8),
            3;
            progress = false,
            chain_type = FlexiChains.VNChain,
            initial_params = InitFromPrior()
        )
        @test chain_h0 isa FlexiChains.VNChain
        pnames = sort(collect(Symbol.(FlexiChains.parameters(chain_h0))))
        @test pnames == [:H0]
        @test all(isfinite, vec(Array(chain_h0[:logjoint])))

        prior_bounds = Dict(
            :H0 => (20.0, 140.0),
            :Ωm => (0.05, 0.95),
            :Ξ₀ => (0.5, 5.0),
            :Ξₙ => (0.05, 3.0),
            :γ => (0.5, 10.0),
            :κ => (0.05, 10.0),
            :zpeak => (0.05, 10.0)
        )
        for (sym, (low, high)) in prior_bounds
            values = vec(Array(chain[sym]))
            @test all(isfinite, values)
            @test all((low .<= values) .& (values .<= high))
        end
        @test all(isfinite, vec(Array(chain[:logjoint])))
    end
end

@testset "submodel boundary lifts VarNames to parent (flat)" begin
    loaded = parity_problem_context(
        :posterior, [Detector("H1"), Detector("L1")]; registry = POPULATION_REGISTRY)
    cache, C, ctx = loaded.problem, loaded.cosmology_type, loaded.ctx
    priors = PARITY_PRIORS

    turing_model = build_turing_model(cache, C, ctx, priors)
    vi = VarInfo(turing_model)
    present = _varinfo_symbols(vi)
    for n in (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
        @test n in present
    end
    @test !(:cosmo_nt in present)
    @test !(:model_nt in present)
end

@testset "condition_turing_model across submodel boundary" begin
    loaded = parity_problem_context(
        :posterior, [Detector("H1"), Detector("L1")]; registry = POPULATION_REGISTRY)
    cache, C, ctx = loaded.problem, loaded.cosmology_type, loaded.ctx
    priors = PARITY_PRIORS
    order = _PARITY_ORDER
    theta0 = PARITY_THETA

    turing_model = build_turing_model(cache, C, ctx, priors)
    @test condition_turing_model(turing_model, theta0, priors, nothing; order = order) ===
          turing_model

    cond_Ξ₀ = condition_turing_model(turing_model, theta0, priors, (:Ξ₀,); order = order)
    sampled = _varinfo_symbols(VarInfo(cond_Ξ₀))
    @test :Ξ₀ in sampled
    for n in (:H0, :Ωm, :Ξₙ, :γ, :κ, :zpeak)
        @test !(n in sampled)
    end

    cond_H0 = condition_turing_model(turing_model, theta0, priors, (:H0,); order = order)
    sampled_H0 = _varinfo_symbols(VarInfo(cond_H0))
    @test :H0 in sampled_H0
    for n in (:Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
        @test !(n in sampled_H0)
    end
end
