# Test-only synthetic importance caches. Include after `using ASGWB` (see `runtests.jl`).
using HDF5

const _PARITY_OBSERVED_SPECTRAL_DENSITY = [0.0, 0.1, 0.2]

const _INTRINSIC_SITE_ORDER_STRINGS = [
    "mass_1_source",
    "mass_2_source",
    "redshift",
    "chi_1",
    "chi_2",
    "lambda_1",
    "lambda_2"
]

const _PARITY_CACHE_ATTRS = (
    command = "ASGWB/test/parity_fixtures.jl (generated test cache)",
    git_revision = "parity-snapshots"
)

function _write_hyperparameters!(hg, hp)
    write(hg, "H0", hp.H0)
    write(hg, "Omega_m", hp.Ωm)
    write(hg, "chi0", hp.Ξ₀)
    write(hg, "chin", hp.Ξₙ)
    return nothing
end

function _write_redshift_spec!(sg, spec; population)
    write(sg, "family", "madau_dickinson")
    write(sg, "z_min", spec.z_min)
    write(sg, "z_max", spec.z_max)
    write(sg, "num_interp", spec.num_interp)
    write(sg, "time_delay_model", "")
    write(sg, "gamma", population.γ)
    write(sg, "kappa", population.κ)
    write(sg, "z_peak", population.zpeak)
    return nothing
end

function _write_proposal_samples!(g, samples)
    attributes(g)[PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR] = PROPOSAL_SAMPLES_SOURCE_TYPE_BNS
    write(g, "mass_1_source", Vector(samples.mass[1, :]))
    write(g, "mass_2_source", Vector(samples.mass[2, :]))
    write(g, "redshift", samples.redshift)
    write(g, "chi_1", samples.χ₁)
    write(g, "chi_2", samples.χ₂)
    write(g, "lambda_1", samples.Λ₁)
    write(g, "lambda_2", samples.Λ₂)
    return nothing
end

function _write_cache_root!(
        f,
        ;
        local_merger_rate::Real,
        observation_time_sec::Real,
        observation_time_yr::Real,
        redshift_integral_fiducial::Union{Real, Nothing} = nothing
)
    a = attributes(f)
    a[IMPORTANCE_CACHE_COMMAND_ATTR] = _PARITY_CACHE_ATTRS.command
    a[IMPORTANCE_CACHE_GIT_REVISION_ATTR] = _PARITY_CACHE_ATTRS.git_revision
    a["local_merger_rate"] = Float64(local_merger_rate)
    a["observation_time_sec"] = Float64(observation_time_sec)
    a["observation_time_yr"] = Float64(observation_time_yr)
    if redshift_integral_fiducial !== nothing
        a["redshift_integral_fiducial"] = Float64(redshift_integral_fiducial)
    end
    return nothing
end

function write_parity_cache_h5(path::AbstractString, variant::Symbol)
    if variant == :posterior
        _write_posterior_cache_h5(path)
    elseif variant == :full_intrinsic
        _write_full_intrinsic_cache_h5(path)
    elseif variant == :importance_context
        _write_importance_context_cache_h5(path)
    elseif variant == :posterior_v2_minimal
        _write_posterior_cache_h5(path)
    else
        throw(ArgumentError("unknown parity cache variant $(repr(variant))"))
    end
    return path
end

function _write_posterior_cache_h5(path::AbstractString)
    frequencies = [10.0, 11.0, 12.0]
    in_band_mask = [false, true, true]
    cached_flux = Float64[1 4; 2 5; 3 6]
    samples = (
        mass = stack_source_masses([1.4, 1.4], [1.2, 1.2]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.0],
        χ₂ = [0.0, 0.0],
        Λ₁ = [100.0, 100.0],
        Λ₂ = [100.0, 100.0]
    )
    intrinsic_vector = Float64[1.4 1.2 0.1 0.0 0.0 100.0 100.0
                               1.4 1.2 0.2 0.0 0.0 100.0 100.0]
    fid = ProposalFiducialParameters(;
        H0 = 67.0,
        Ωm = 0.315,
        Ξ₀ = 1.0,
        Ξₙ = 0.0,
        γ = 2.7,
        κ = 5.7,
        zpeak = 2.0
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 64, nothing)
    h5open(path, "w") do f
        _write_cache_root!(
            f;
            local_merger_rate = 1e-7,
            observation_time_sec = 2.0,
            observation_time_yr = 1e-6
        )
        write(f, "intrinsic_site_order", _INTRINSIC_SITE_ORDER_STRINGS)
        write(f, "proposal_intrinsic_vector", Matrix(permutedims(intrinsic_vector)))
        write(f, "frequencies", frequencies)
        write(f, "in_band_mask", in_band_mask)
        write(f, "cached_flux", cached_flux)
        write(f, "fiducial_spectral_density", _PARITY_OBSERVED_SPECTRAL_DENSITY)
        _write_proposal_samples!(create_group(f, "proposal_samples"), samples)
        hg = create_group(f, "hyperparameters")
        _write_hyperparameters!(hg, fid)
        write(hg, "gamma", fid.γ)
        write(hg, "kappa", fid.κ)
        write(hg, "z_peak", fid.zpeak)
        sg = create_group(f, "redshift_prior_spec")
        _write_redshift_spec!(sg, spec; population = fid)
    end
    return path
end

function _write_full_intrinsic_cache_h5(path::AbstractString)
    frequencies = [10.0, 11.0, 12.0]
    in_band_mask = [false, true, true]
    cached_flux = Float64[1.0 1.5 2.0 2.5; 2.0 2.5 3.0 3.5; 3.0 3.5 4.0 4.5]
    samples = (
        mass = stack_source_masses([1.8, 2.2, 1.4, 2.4], [1.2, 1.7, 1.1, 1.3]),
        redshift = [0.1, 0.2, 0.3, 0.5],
        χ₁ = [0.0, 0.2, -0.1, 0.5],
        χ₂ = [0.1, -0.2, 0.0, 0.3],
        Λ₁ = [400.0, 800.0, 1200.0, 2000.0],
        Λ₂ = [300.0, 600.0, 700.0, 1500.0]
    )
    intrinsic_vector = Float64[1.8 2.2 1.4 2.4
                               1.2 1.7 1.1 1.3
                               0.1 0.2 0.3 0.5
                               0.0 0.2 -0.1 0.5
                               0.1 -0.2 0.0 0.3
                               400.0 800.0 1200.0 2000.0
                               300.0 600.0 700.0 1500.0]
    fid = ProposalFiducialParameters(;
        H0 = 67.0,
        Ωm = 0.315,
        Ξ₀ = 1.0,
        Ξₙ = 0.0,
        γ = 2.7,
        κ = 5.7,
        zpeak = 2.0
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 64, nothing)
    h5open(path, "w") do f
        _write_cache_root!(
            f;
            local_merger_rate = 1e-7,
            observation_time_sec = 2.0,
            observation_time_yr = 1e-6
        )
        write(f, "intrinsic_site_order", _INTRINSIC_SITE_ORDER_STRINGS)
        write(f, "proposal_intrinsic_vector", Matrix(permutedims(intrinsic_vector)))
        write(f, "frequencies", frequencies)
        write(f, "in_band_mask", in_band_mask)
        write(f, "cached_flux", cached_flux)
        write(f, "fiducial_spectral_density", _PARITY_OBSERVED_SPECTRAL_DENSITY)
        _write_proposal_samples!(create_group(f, "proposal_samples"), samples)
        hg = create_group(f, "hyperparameters")
        _write_hyperparameters!(hg, fid)
        sg = create_group(f, "redshift_prior_spec")
        _write_redshift_spec!(sg, spec; population = fid)
    end
    return path
end

function _write_importance_context_cache_h5(path::AbstractString)
    frequencies = [1.0, 2.0]
    in_band_mask = [true, true]
    cached_flux = Float64[1.0 1.5; 2.0 2.5]
    samples = (
        mass = stack_source_masses([1.4, 1.4], [1.2, 1.2]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.0],
        χ₂ = [0.0, 0.0],
        Λ₁ = [100.0, 100.0],
        Λ₂ = [100.0, 100.0]
    )
    intrinsic_vector = Float64[1.4 1.2 0.1 0.0 0.0 100.0 100.0
                               1.4 1.2 0.2 0.0 0.0 100.0 100.0]
    fid = ProposalFiducialParameters(;
        H0 = 67.0,
        Ωm = 0.315,
        Ξ₀ = 1.0,
        Ξₙ = 0.0,
        γ = 2.7,
        κ = 3.0,
        zpeak = 2.5
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 1024, nothing)
    h5open(path, "w") do f
        _write_cache_root!(
            f;
            local_merger_rate = 161.0,
            observation_time_sec = 3.15576e7,
            observation_time_yr = 1.0,
            redshift_integral_fiducial = 1.0
        )
        write(f, "intrinsic_site_order", _INTRINSIC_SITE_ORDER_STRINGS)
        write(f, "proposal_intrinsic_vector", Matrix(permutedims(intrinsic_vector)))
        write(f, "frequencies", frequencies)
        write(f, "in_band_mask", in_band_mask)
        write(f, "cached_flux", cached_flux)
        write(f, "fiducial_spectral_density", [0.0, 0.0])
        _write_proposal_samples!(create_group(f, "proposal_samples"), samples)
        hg = create_group(f, "hyperparameters")
        _write_hyperparameters!(hg, fid)
        write(hg, "gamma", fid.γ)
        write(hg, "kappa", fid.κ)
        write(hg, "z_peak", fid.zpeak)
        sg = create_group(f, "redshift_prior_spec")
        _write_redshift_spec!(sg, spec; population = fid)
    end
    return path
end

const _PARITY_CACHE_PATHS = Dict{Symbol, String}()

function parity_cache_path(variant::Symbol)
    get(_PARITY_CACHE_PATHS, variant) do
        path = joinpath(Base.mktempdir(), "asgwb_parity_$(variant).h5")
        write_parity_cache_h5(path, variant)
        _PARITY_CACHE_PATHS[variant] = path
        return path
    end
end

function resolve_parity_cache_path(path::AbstractString)
    if path == "parity:posterior"
        return parity_cache_path(:posterior)
    elseif path == "parity:full_intrinsic"
        return parity_cache_path(:full_intrinsic)
    elseif path == "parity:importance_context"
        return parity_cache_path(:importance_context)
    elseif path == "parity:posterior_v2_minimal"
        return parity_cache_path(:posterior_v2_minimal)
    end
    return path
end
