using HDF5
using LogDensityProblems
using Test

@testset "AdvancedHMC smoke test" begin
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

            problem = ASGWBLogDensity(cache, priors)
            z0 = unconstrained_initial_point(problem, theta0)
            ad_problem = ad_logdensity(problem)
            logdensity,
            gradient = LogDensityProblems.logdensity_and_gradient(ad_problem, z0)
            reference_logdensity,
            reference_gradient = finite_difference_logdensity_and_gradient(problem, z0)

            @test LogDensityProblems.dimension(problem) == 7
            @test LogDensityProblems.dimension(ad_problem) == 7
            @test isfinite(logdensity)
            @test all(isfinite, gradient)
            @test logdensity ≈ reference_logdensity rtol = 1e-9
            # Loose tolerance: log-density magnitudes are large; detector-rebuilt effective_psd
            # can leave AD vs finite differences slightly misaligned on some grids.
            @test gradient ≈ reference_gradient rtol = 0.05

            samples, stats,
            sampling_problem = sample_with_advancedhmc(
                cache, priors, theta0; n_adapts = 3, n_samples = 3)

            @test sampling_problem isa ASGWBLogDensity
            @test length(samples) == 3
            @test length(stats) == 3

            _prop_sym = Dict(
                "H0" => :H0,
                "Omega_m" => :Ωm,
                "chi0" => :Ξ₀,
                "chin" => :Ξₙ,
                "gamma" => :γ,
                "kappa" => :κ,
                "z_peak" => :zpeak
            )
            for sample in samples
                for (name, (low, high)) in prior_bounds
                    value = getproperty(sample, _prop_sym[name])
                    @test isfinite(value)
                    @test low <= value <= high
                end
            end
        end
    end
end
