#!/usr/bin/env julia
# Refresh `test/fixtures/deterministic_parity.h5` posterior_case from the posterior
# importance cache (must match the HDF5 layout consumed by `load_cache`).
#
# Run from the package root:
#   julia --project=. contrib/migrate_fixtures_full_bns.jl
#
# Historical redshift-only → full-BNS migration lived here; fixtures are now shipped
# in the prototype layout (`contrib/upgrade_hdf5_importance_caches.jl`).
using ASGWB
using HDF5

function refresh_posterior_case_parity!(parity_path::AbstractString, cache_path::AbstractString)
    cache = load_cache(cache_path, [Detector("H1"), Detector("L1")])
    h5open(parity_path, "r+") do f
        g = f["posterior_case"]
        theta = HyperParameters((; (
            Symbol(name) => Float64(read(g["theta/$(name)"])) for
            name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
        )...,))
        priors = build_uniform_priors(
            Dict(
                name => (
                    Float64(read(g["prior_bounds/$(name)/low"])),
                    Float64(read(g["prior_bounds/$(name)/high"])),
                ) for
                name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
            ),
        )
        ev = evaluate_importance_terms(theta, cache)
        function _overwrite!(grp, name, data)
            haskey(grp, name) && HDF5.delete_object(grp, name)
            write(grp, name, data)
        end
        _overwrite!(g, "dgw_theta_sq", collect(ev.dgw_theta_sq))
        _overwrite!(g, "weights", collect(ev.weights))
        _overwrite!(g, "spectral_density_full", collect(ev.spectral_density))
        _overwrite!(g, "spectral_density_in_band", collect(ev.spectral_density_in_band))
        _overwrite!(g, "expected_number_of_sources", ev.expected_number_of_sources)
        _overwrite!(g, "log_ratio", collect(ev.log_ratio))
        _overwrite!(g, "target_log_prob", collect(ev.target_log_prob))
        _overwrite!(g, "redshift_integral", ev.redshift_integral)
        _overwrite!(g, "log_prior", logprior(theta, priors))
        _overwrite!(g, "log_likelihood", loglikelihood(theta, cache))
        _overwrite!(g, "log_posterior", logposterior(theta, cache, priors))
        _overwrite!(g, "normalized_ess", normalized_ess(ev.weights))
        _overwrite!(g, "max_normalized_weight", max_normalized_weight(ev.weights))
        _overwrite!(g, "log_ratio_variance", log_ratio_variance(ev.log_ratio))
    end
end

root = joinpath(@__DIR__, "..")
fixtures = joinpath(root, "test", "fixtures")
par = joinpath(fixtures, "deterministic_parity.h5")
function refresh_full_intrinsic_case_parity!(parity_path::AbstractString, cache_path::AbstractString)
    cache = load_cache(cache_path, [Detector("H1"), Detector("L1")])
    h5open(parity_path, "r+") do f
        g = f["full_intrinsic_case"]
        theta = HyperParameters((; (
            Symbol(name) => Float64(read(g["theta/$(name)"])) for
            name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
        )...,))
        priors = build_uniform_priors(
            Dict(
                name => (
                    Float64(read(g["prior_bounds/$(name)/low"])),
                    Float64(read(g["prior_bounds/$(name)/high"])),
                ) for
                name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
            ),
        )
        ev = evaluate_importance_terms(theta, cache)
        function _overwrite!(grp, name, data)
            haskey(grp, name) && HDF5.delete_object(grp, name)
            write(grp, name, data)
        end
        _overwrite!(g, "dgw_theta_sq", collect(ev.dgw_theta_sq))
        _overwrite!(g, "weights", collect(ev.weights))
        _overwrite!(g, "spectral_density_full", collect(ev.spectral_density))
        _overwrite!(g, "expected_number_of_sources", ev.expected_number_of_sources)
        _overwrite!(g, "log_ratio", collect(ev.log_ratio))
        _overwrite!(g, "target_log_prob", collect(ev.target_log_prob))
        _overwrite!(g, "redshift_integral", ev.redshift_integral)
        _overwrite!(g, "log_prior", logprior(theta, priors))
        _overwrite!(g, "log_likelihood", loglikelihood(theta, cache))
        _overwrite!(g, "log_posterior", logposterior(theta, cache, priors))
    end
end

refresh_posterior_case_parity!(par, joinpath(fixtures, "posterior_cache_julia.h5"))
println("refreshed deterministic_parity.h5 posterior_case")
refresh_full_intrinsic_case_parity!(par, joinpath(fixtures, "full_intrinsic_cache_julia.h5"))
println("refreshed deterministic_parity.h5 full_intrinsic_case")
