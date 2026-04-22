using HDF5
using Test
using Turing

@testset "Turing model parity and smoke test" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    h5open(fixture_path, "r") do file
        for (cache_filename, group_name) in (
            ("posterior_cache_julia.h5", "posterior_case"),
            ("full_intrinsic_cache_julia.h5", "full_intrinsic_case")
        )
            cache = load_cache(
                joinpath(@__DIR__, "fixtures", cache_filename),
                [Detector("H1"), Detector("L1")]
            )
            group = file[group_name]

            theta0 = HyperParameters(;
                H0 = Float64(read(group["theta/H0"])),
                Ωm = Float64(read(group["theta/Omega_m"])),
                Ξ₀ = Float64(read(group["theta/chi0"])),
                Ξₙ = Float64(read(group["theta/chin"])),
                γ = Float64(read(group["theta/gamma"])),
                κ = Float64(read(group["theta/kappa"])),
                zpeak = Float64(read(group["theta/z_peak"]))
            )
            prior_bounds = Dict(
                name => (
                    Float64(read(group["prior_bounds/$(name)/low"])),
                    Float64(read(group["prior_bounds/$(name)/high"]))
                ) for
            name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
            )
            priors = build_uniform_priors(prior_bounds)

            model = build_turing_model(cache, priors; track = false)
            @test Turing.logjoint(model, theta0) ≈ logposterior(theta0, cache, priors) rtol = 1e-6

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

            chain,
            sampled_model = sample_with_turing(
                cache, priors, theta0; n_adapts = 3, n_samples = 3, track = false)

            @test Turing.logjoint(sampled_model, theta0) ≈ Turing.logjoint(model, theta0) rtol = 1e-6
            @test size(chain, 1) == 3

            chain_h0,
            cond_h0 = sample_with_turing(
                cache,
                priors,
                theta0;
                n_adapts = 3,
                n_samples = 3,
                track = false,
                sample_only = (:H0,)
            )
            pnames = sort(collect(Symbol.(Turing.MCMCChains.names(chain_h0, :parameters))))
            @test pnames == [:H0]
            @test all(isfinite, vec(Array(chain_h0[:, :logjoint, :])))

            _chain_sym = Dict(
                "H0" => :H0,
                "Omega_m" => :Ωm,
                "chi0" => :Ξ₀,
                "chin" => :Ξₙ,
                "gamma" => :γ,
                "kappa" => :κ,
                "z_peak" => :zpeak,
            )
            for (name, (low, high)) in prior_bounds
                values = vec(Array(chain[:, _chain_sym[name], :]))
                @test all(isfinite, values)
                @test all((low .<= values) .& (values .<= high))
            end

            @test all(isfinite, vec(Array(chain[:, :logjoint, :])))
        end
    end
end
