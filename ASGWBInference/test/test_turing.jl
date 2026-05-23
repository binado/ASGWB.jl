using Test
using Turing
using FlexiChains
using ASGWB
using ASGWBInference: build_turing_model, condition_turing_model

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

        model = build_turing_model(cache, priors; track = false)
        @test Turing.logjoint(model, theta0) ≈ logposterior(theta0, cache, priors) rtol = 1e-6
        @test condition_turing_model(model, theta0, priors, nothing) === model
        @test_throws ArgumentError condition_turing_model(model, theta0, priors, ())
        @test_throws ArgumentError condition_turing_model(model, theta0, priors, (:unknown,))

        model_track = build_turing_model(cache, priors; track = true)
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
        sampled_model = condition_turing_model(model, theta0, priors, nothing)
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

        cond_h0 = condition_turing_model(model, theta0, priors, (:H0,))
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
