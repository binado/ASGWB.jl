using HDF5
using Test
using Turing

@testset "Turing model parity and smoke test" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    h5open(fixture_path, "r") do file
        for (cache_filename, group_name) in (
            ("posterior_cache_julia.h5", "posterior_case"),
            ("full_intrinsic_cache_julia.h5", "full_intrinsic_case"),
        )
            cache = load_cache(
                joinpath(@__DIR__, "fixtures", cache_filename),
                [Detector("H1"), Detector("L1")],
            )
            group = file[group_name]

            theta0 = HyperParameters((; (
                Symbol(name) => Float64(read(group["theta/$(name)"])) for
                name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
            )...,))
            prior_bounds = Dict(
                name => (
                    Float64(read(group["prior_bounds/$(name)/low"])),
                    Float64(read(group["prior_bounds/$(name)/high"])),
                ) for
                name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
            )
            priors = build_uniform_priors(prior_bounds)

            model = build_turing_model(cache, priors)
            theta_flat = as_flat_constrained(theta0)
            @test Turing.logjoint(model, theta_flat) ≈
                logposterior(theta0, cache, priors) rtol = 1e-6

            chain, sampled_model =
                sample_with_turing(cache, priors, theta0; n_adapts=3, n_samples=3)

            @test Turing.logjoint(sampled_model, theta_flat) ≈
                Turing.logjoint(model, theta_flat) rtol = 1e-6
            @test size(chain, 1) == 3

            for (name, (low, high)) in prior_bounds
                values = vec(Array(chain[:, Symbol(name), :]))
                @test all(isfinite, values)
                @test all((low .<= values) .& (values .<= high))
            end

            @test all(isfinite, vec(Array(chain[:, :logjoint, :])))
        end
    end
end
