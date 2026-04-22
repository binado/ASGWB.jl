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

            theta0 = HyperParameters((;
                (
                Symbol(name) => Float64(read(group["theta/$(name)"]))
            for name in
                ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
            )...,
            ))
            prior_bounds = Dict(
                name => (
                    Float64(read(group["prior_bounds/$(name)/low"])),
                    Float64(read(group["prior_bounds/$(name)/high"]))
                ) for
            name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
            )
            priors = build_uniform_priors(prior_bounds)

            model = build_turing_model(cache, priors)
            @test Turing.logjoint(model, theta0) ≈ logposterior(theta0, cache, priors) rtol = 1e-6

            returned_nt = Turing.returned(model, theta0)
            @test haskey(returned_nt, :effective_sample_size)
            @test isfinite(returned_nt.effective_sample_size)
            @test 0 < returned_nt.effective_sample_size <= 1

            chain,
            sampled_model = sample_with_turing(
                cache, priors, theta0; n_adapts = 3, n_samples = 3)

            @test Turing.logjoint(sampled_model, theta0) ≈ Turing.logjoint(model, theta0) rtol = 1e-6
            @test size(chain, 1) == 3

            chain_h0,
            cond_h0 = sample_with_turing(
                cache,
                priors,
                theta0;
                n_adapts = 3,
                n_samples = 3,
                sample_only = (:H0,)
            )
            pnames = sort(collect(Symbol.(Turing.MCMCChains.names(chain_h0, :parameters))))
            @test pnames == [:H0]
            @test all(isfinite, vec(Array(chain_h0[:, :logjoint, :])))

            for (name, (low, high)) in prior_bounds
                values = vec(Array(chain[:, Symbol(name), :]))
                @test all(isfinite, values)
                @test all((low .<= values) .& (values .<= high))
            end

            @test all(isfinite, vec(Array(chain[:, :logjoint, :])))
        end
    end
end
