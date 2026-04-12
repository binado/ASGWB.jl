#!/usr/bin/env julia
# Rewrite committed HDF5 importance caches to the prototype layout (provenance attrs,
# cached_flux, no on-disk covariance / sgwb_scale / format_*).
#
# Run from the package root:
#   julia --project=. contrib/upgrade_hdf5_importance_caches.jl

using ASGWB
using HDF5
using HDF5: delete_object

const _FIXTURE_COMMAND =
    "contrib/upgrade_hdf5_importance_caches.jl (ASGWB.jl committed test fixtures)"
const _FIXTURE_GIT_REVISION = "fixture-hdf5-refresh"

function _as_string(x)
    x isa AbstractString && return String(x)
    x isa AbstractVector{UInt8} && return String(copy(x))
    return string(x)
end

function _optional_float(g, k)
    haskey(g, k) || return nothing
    return Float64(read(g[k]))
end

function _fid_from_groups(hg, sg)::ProposalFiducialParameters
    family = parse_redshift_prior_family(_as_string(read(sg["family"])))
    H0 = Float64(read(hg["H0"]))
    Omega_m = Float64(read(hg["Omega_m"]))
    chi0 = Float64(read(hg["chi0"]))
    chin = Float64(read(hg["chin"]))
    γ = _optional_float(hg, "gamma")
    κ = _optional_float(hg, "kappa")
    zp = _optional_float(hg, "z_peak")
    λ = _optional_float(hg, "lamb")
    γs = _optional_float(sg, "gamma")
    κs = _optional_float(sg, "kappa")
    zps = _optional_float(sg, "z_peak")
    λs = _optional_float(sg, "lamb")
    function pick(h, s, name)
        h !== nothing && s !== nothing && h != s && throw(ArgumentError("mismatch $name"))
        return something(h, s)
    end
    if family == MadauDickinson
        γ = pick(γ, γs, "gamma")
        κ = pick(κ, κs, "kappa")
        zp = pick(zp, zps, "z_peak")
        γ === nothing && throw(ArgumentError("missing gamma"))
        κ === nothing && throw(ArgumentError("missing kappa"))
        zp === nothing && throw(ArgumentError("missing z_peak"))
        return ProposalFiducialParameters(; H0, Omega_m, chi0, chin, gamma=γ, kappa=κ, z_peak=zp)
    end
    λ = pick(λ, λs, "lamb")
    λ === nothing && throw(ArgumentError("missing lamb"))
    return ProposalFiducialParameters(; H0, Omega_m, chi0, chin, lamb=λ)
end

function _upgrade_one(path::AbstractString)
    h5open(path, "r+") do f
        attrs = attributes(f)
        for name in ("format_name", "format_version")
            haskey(attrs, name) && HDF5.delete_attribute(f, name)
        end
        for name in (IMPORTANCE_CACHE_COMMAND_ATTR, IMPORTANCE_CACHE_GIT_REVISION_ATTR)
            haskey(attrs, name) && HDF5.delete_attribute(f, name)
        end
        attrs[IMPORTANCE_CACHE_COMMAND_ATTR] = _FIXTURE_COMMAND
        attrs[IMPORTANCE_CACHE_GIT_REVISION_ATTR] = _FIXTURE_GIT_REVISION

        if haskey(f, "cached_flux_over_dgw2")
            fid = _fid_from_groups(f["hyperparameters"], f["redshift_prior_spec"])
            F = Matrix{Float64}(
                permutedims(Array{Float64}(read(f["cached_flux_over_dgw2"]))),
            )
            z = vec(Float64.(read(f["proposal_samples/redshift"])))
            d_l = luminosity_distance.(z, fid.H0, fid.Omega_m)
            d_gw = gravitational_wave_distance.(z, d_l, fid.chi0, fid.chin)
            scale = Float64.((d_l ./ d_gw) .^ 2)
            cached = F ./ reshape(scale, :, 1)
            delete_object(f, "cached_flux_over_dgw2")
            write(f, "cached_flux", Matrix(permutedims(cached)))
        end

        for name in ("covariance", "sgwb_scale", "proposal_log_prob", "dgw_fid_sq")
            haskey(f, name) && delete_object(f, name)
        end

        g = f["proposal_samples"]
        ag = attributes(g)
        if haskey(ag, PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR)
            HDF5.delete_attribute(g, PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR)
        end
        ag[PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR] = PROPOSAL_SAMPLES_SOURCE_TYPE_BNS
    end
    println("upgraded ", path)
end

function main()
    root = joinpath(@__DIR__, "..", "test", "fixtures")
    for name in (
        "importance_context_julia.h5",
        "posterior_cache_julia.h5",
        "posterior_cache_julia_v2_minimal.h5",
        "full_intrinsic_cache_julia.h5",
    )
        p = joinpath(root, name)
        isfile(p) || error("missing $p")
        _upgrade_one(p)
    end
end

main()
