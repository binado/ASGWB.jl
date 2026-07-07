### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 65017378-13ca-4818-a3f9-2683e7a3aad4
md"""
# Fiducial SNR across detector networks

This notebook evaluates the astrophysical gravitational-wave background spectral density
``S_h(f)`` once, at fiducial hyperparameter values, and then reports the matched-filter
SNR for each detector network configuration swept over by `scripts/generate_mcmc_configs.jl`.

``S_h(f)`` depends only on the signal (fluxes, merger rate, importance weights) and not on
the detector network, so it is computed a single time and reused across all networks; only
the network's effective PSD changes per row.
"""

# ╔═╡ 7ce31254-65ab-4314-8bda-6233736a9a5e
begin
    _repo_root = normpath(joinpath(@__DIR__, ".."))
    catalog_path = joinpath(_repo_root, "catalog.h5")

    seed = 42
    @info "seeding RNG" rng_seed = seed
    Random.seed!(seed)

    local_merger_rate = 161.0 # Matches COBA simulations
    observation_time = 1.0

    cosmology_parameters = (;
        H0 = 67.66,
        Ωm = 0.3096,
        w0 = -1,
        Ξ₀ = 1.0,
        Ξₙ = 1.91
    )
    fiducials = (;
        cosmology_parameters...,
        γ = 2.7,
        κ = 3.0,
        zpeak = 2.0
    )

    # Defining cosmology and propagation. Background expansion `C` and GW propagation `P`
    # are orthogonal axes (use `GR` for standard propagation).
    C = W0CDM
    P = ModifiedPropagation
end

# ╔═╡ b56cde8c-c6d9-4f08-ae24-d355b656405d
begin
    @info "loading catalog" catalog_path
    loaded = load_catalog(catalog_path)
    catalog = loaded.catalog

    samples = bns_samples_from_catalog(catalog.samples, C, fiducials)
    model = prepare_bns_madau_dickinson_model(
        samples,
        fiducials,
        C,
        P;
        observation_time = observation_time,
        local_merger_rate = local_merger_rate
    )

    f = frequencies(loaded.metadata.grid)
    mask = in_band_mask(loaded.metadata.grid)
    fluxes = catalog.fluxes

    @info "catalog loaded" n_frequency_bins=length(f) n_proposal_samples=length(
        samples.redshift,
    )
end

# ╔═╡ cb6787ae-d5dc-41c9-8043-dc6d176a1f48
md"""
## Visualizing ``\Omega_{\mathrm{GW}}``

We plot ``\Omega_{\mathrm{GW}}(f)`` for the fiducial values of the parameters ``\Lambda``,
alongside the SNR for a representative detector network.
"""

# ╔═╡ eaeca0fc-abc8-423e-94e6-a658a3e40580
begin
    rate0, log_weights0 = merger_rate_and_log_weights(model, fiducials, samples)
    Sh0 = spectral_density(fluxes, rate0; weights = exp.(log_weights0))
end

# ╔═╡ 6539c3ae-fac4-4ad4-a8d2-0b41ea6b00db
function plot_fiducial_omega_gw(Sh0, f, fiducials, observation)
    df = frequency_bin_width(f)
    snr = spectral_snr(
        Sh0,
        observation.effective_psd,
        year_to_second(observation.observation_time),
        df
    )

    Ωgw_plot = Ωgw(Sh0, f, fiducials.H0)
    mask = Ωgw_plot .> 0.0
    fm = f[mask]
    Ωgw_pos = Ωgw_plot[mask]
    fig = Figure(size = (900, 450))
    ax = Axis(
        fig[1, 1];
        xlabel = L"$f~\mathrm{(Hz)}$",
        ylabel = L"$\Omega_{\mathrm{GW}}(f)$",
        xscale = log10,
        yscale = log10,
        limits = (nothing, nothing, 1e-15, nothing)
    )
    if !isempty(Ωgw_pos)
        label = @sprintf "SNR = %.1f" snr
        lines!(ax, fm, Ωgw_pos; label = label)
        axislegend(ax; position = :rt)
    end
    return fig
end

# ╔═╡ 33387fc6-9ee1-4194-b5b8-276da883501b
begin
    detnames = [:S1, :R1, :C1]
    detectors = Detector.(string.(detnames))
    observation = build_observation_context(f, detectors, mask, observation_time)
    plot_fiducial_omega_gw(Sh0, f, fiducials, observation)
end

# ╔═╡ 7acea2e1-d6d8-430c-b613-be08b57677d1
md"""
## SNR per detector network

Compare the fiducial matched-filter SNR across the detector network configurations swept
over by `scripts/generate_mcmc_configs.jl`. `Sh0` is detector-independent and was computed
once above; only each network's effective PSD changes per row.
"""

# ╔═╡ a3adb7e1-5222-4d1f-b947-0c9639992ee3
DETECTOR_NETWORKS = (
    "ET-triangular" => ["E1", "E2", "E3"],
    "ET-triangular-CE-Hanford" => ["E1", "E2", "E3", "C1"],
    "ET-2L-aligned" => ["S1", "R1"],
    "ET-2L-aligned-CE-Hanford" => ["S1", "R1", "C1"],
    "ET-2L-misaligned" => ["S2", "R2"],
    "ET-2L-misaligned-CE-Hanford" => ["S2", "R2", "C1"]
)

# ╔═╡ e74fb94f-4d33-47bc-8f92-0bbeff19a8a7
begin
    df_bin = frequency_bin_width(f)
    T_sec = year_to_second(observation_time)
    rows = map(DETECTOR_NETWORKS) do (label, detstrs)
        dets = Detector.(detstrs)
        obs = build_observation_context(f, dets, mask, observation_time)
        snr = spectral_snr(Sh0, obs.effective_psd, T_sec, df_bin)
        (network = label, detectors = join(detstrs, ","), SNR = snr)
    end
    snr_table = DataFrame(rows)
end

# ╔═╡ e3451b84-3c00-4445-b712-d49139a62346
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    using AstroSGWB
    using AstroSGWB:
                     Detector,
                     frequencies,
                     in_band_mask,
                     build_observation_context,
                     load_catalog,
                     W0CDM,
                     ModifiedPropagation,
                     spectral_density,
                     spectral_snr,
                     frequency_bin_width,
                     year_to_second,
                     Ωgw
    using AstroSGWBImportanceModels:
                                     bns_samples_from_catalog,
                                     prepare_bns_madau_dickinson_model
    using AstroSGWBInference: merger_rate_and_log_weights
    using Random
    using CairoMakie
    using LaTeXStrings
    using Printf
    using DataFrames
end

# ╔═╡ Cell order:
# ╠═65017378-13ca-4818-a3f9-2683e7a3aad4
# ╠═e3451b84-3c00-4445-b712-d49139a62346
# ╠═7ce31254-65ab-4314-8bda-6233736a9a5e
# ╠═b56cde8c-c6d9-4f08-ae24-d355b656405d
# ╟─cb6787ae-d5dc-41c9-8043-dc6d176a1f48
# ╠═eaeca0fc-abc8-423e-94e6-a658a3e40580
# ╠═6539c3ae-fac4-4ad4-a8d2-0b41ea6b00db
# ╠═33387fc6-9ee1-4194-b5b8-276da883501b
# ╟─7acea2e1-d6d8-430c-b613-be08b57677d1
# ╠═a3adb7e1-5222-4d1f-b947-0c9639992ee3
# ╠═e74fb94f-4d33-47bc-8f92-0bbeff19a8a7
