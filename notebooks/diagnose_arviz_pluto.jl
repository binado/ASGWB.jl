### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 3a065958-b6f1-4855-ad59-803892b592de
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()

    using ArviZ
    using ArviZPythonPlots
    using NCDatasets
    using Statistics
end

# ╔═╡ 55bb7b84-631d-4c9a-9295-3ef239427d75
md"""
# ArviZ NetCDF diagnostics

Load a saved ArviZ `InferenceData` NetCDF file and inspect MCMC diagnostics without
running a sampler. Edit only `netcdf_path` when switching chain files.
"""

# ╔═╡ c66dee78-6a7e-4891-b90c-85959e2638b7
netcdf_path = joinpath(@__DIR__, "..", "chains-H0-γ-κ-zpeak.nc")

# ╔═╡ ef3f89c0-e204-4141-a985-26649d598d9e
begin
    isfile(netcdf_path) ||
        throw(ArgumentError("NetCDF file not found: $(repr(netcdf_path))"))
    idata = from_netcdf(netcdf_path)
end

# ╔═╡ e126abe3-591b-4143-a5bf-2af3390136c5
begin
    posterior_var_symbols = collect(propertynames(idata.posterior))
    posterior_var_names = string.(posterior_var_symbols)
end

# ╔═╡ 52ce672e-78eb-464f-9435-40d33ad2b1e0
selected_var_names = posterior_var_names

# ╔═╡ 7871eec2-3894-4ae3-981d-0c0a22cfb5fe
begin
    function has_var(dataset, name::Symbol)
        return name in propertynames(dataset)
    end

    function variable_array(dataset, name::Symbol)
        return Array(getproperty(dataset, name))
    end

    function numeric_summary(x)
        values = collect(skipmissing(vec(Float64.(Array(x)))))
        isempty(values) &&
            return (mean = missing, min = missing, median = missing, max = missing)
        return (
            mean = mean(values),
            min = minimum(values),
            median = median(values),
            max = maximum(values)
        )
    end

    function sample_stats_diagnostics(idata)
        hasproperty(idata, :sample_stats) || return "sample_stats group not present"
        stats = idata.sample_stats
        return (
            variables = propertynames(stats),
            divergence_count = has_var(stats, :diverging) ?
                               count(!iszero, vec(variable_array(stats, :diverging))) :
                               missing,
            acceptance_rate = has_var(stats, :acceptance_rate) ?
                              numeric_summary(variable_array(stats, :acceptance_rate)) :
                              missing,
            tree_depth = has_var(stats, :tree_depth) ?
                         numeric_summary(variable_array(stats, :tree_depth)) : missing,
            n_steps = has_var(stats, :n_steps) ?
                      numeric_summary(variable_array(stats, :n_steps)) : missing,
            bfmi = has_var(stats, :energy) ?
                   bfmi(variable_array(stats, :energy); dims = 1) : missing
        )
    end

    nothing
end

# ╔═╡ a8dc8fc0-486a-4395-852b-857fb12e37d6
md"""
## Data
"""

# ╔═╡ 704aecb0-0157-4dae-b9b4-471489be1758
propertynames(idata)

# ╔═╡ 1e617dd3-8749-44cc-a3a5-25f49a5e4023
posterior_var_names

# ╔═╡ abf9ab52-3622-4106-b465-b6542b090040
md"""
## Numeric diagnostics
"""

# ╔═╡ f08e0893-fea6-4e86-9c7a-0287d9375e71
summarize(idata; group = :posterior)

# ╔═╡ 7b0a9e36-1df6-4eae-b740-32b5b40f342b
ess(idata.posterior; kind = :bulk)

# ╔═╡ cb0d7f30-f769-43d8-b863-88638fc4f44f
ess(idata.posterior; kind = :tail)

# ╔═╡ 92e13255-b5e8-4127-afdc-978ec57f686f
rhat(idata.posterior; kind = :rank)

# ╔═╡ b5e3fdd5-c9cf-4c1b-8d1f-ffe60d663fee
mcse(idata.posterior)

# ╔═╡ 2d66b01a-5be9-452c-b9e3-dcd8a760db48
sample_stats_diagnostics(idata)

# ╔═╡ f4408d42-71ab-43a2-94fd-e7f9df15fbb7
md"""
## Trace and rank plots
"""

# ╔═╡ 18ae0915-f9c1-4564-b702-962e4ac897d0
plot_trace(idata; var_names = selected_var_names)

# ╔═╡ 8a77505c-8be1-48e3-9c86-accd8368f571
plot_rank(idata; var_names = selected_var_names)

# ╔═╡ a426dc7d-06b9-499b-88aa-7cf8be1b99f4
md"""
## ESS plots
"""

# ╔═╡ 335ab9b1-e7be-46af-a6a6-3171adfda0d1
plot_ess(idata; var_names = selected_var_names, kind = "local")

# ╔═╡ 3ca2874c-2596-4b5d-8edb-af231023feaf
plot_ess(idata; var_names = selected_var_names, kind = "quantile")

# ╔═╡ 98c4315a-c235-4881-9ca6-95e2c8e49bcb
plot_ess(idata; var_names = selected_var_names, kind = "evolution")

# ╔═╡ ec517be9-2426-484f-86ba-4b339bb3db00
md"""
## Autocorrelation and summary plots
"""

# ╔═╡ 3e3f14f7-51ad-42fe-95a0-d0eea4c8623b
plot_autocorr(idata; var_names = selected_var_names)

# ╔═╡ b7096208-eb1a-4b26-9337-a9f66a6f9682
plot_forest(idata; var_names = selected_var_names, ess = true, r_hat = true)

# ╔═╡ d08a483d-4823-4cff-9d68-89a29005e51a
begin
    if length(selected_var_names) >= 2
        plot_pair(idata; var_names = selected_var_names)
    else
        plot_posterior(idata; var_names = selected_var_names)
    end
end

# ╔═╡ 3135ae24-c5c3-41cb-ad78-f2eec786ee8e
begin
    if hasproperty(idata, :sample_stats) && has_var(idata.sample_stats, :energy)
        plot_energy(idata)
    else
        "sample_stats.energy not present"
    end
end

# ╔═╡ Cell order:
# ╠═3a065958-b6f1-4855-ad59-803892b592de
# ╟─55bb7b84-631d-4c9a-9295-3ef239427d75
# ╠═c66dee78-6a7e-4891-b90c-85959e2638b7
# ╠═ef3f89c0-e204-4141-a985-26649d598d9e
# ╠═e126abe3-591b-4143-a5bf-2af3390136c5
# ╠═52ce672e-78eb-464f-9435-40d33ad2b1e0
# ╠═7871eec2-3894-4ae3-981d-0c0a22cfb5fe
# ╟─a8dc8fc0-486a-4395-852b-857fb12e37d6
# ╠═704aecb0-0157-4dae-b9b4-471489be1758
# ╠═1e617dd3-8749-44cc-a3a5-25f49a5e4023
# ╟─abf9ab52-3622-4106-b465-b6542b090040
# ╠═f08e0893-fea6-4e86-9c7a-0287d9375e71
# ╠═7b0a9e36-1df6-4eae-b740-32b5b40f342b
# ╠═cb0d7f30-f769-43d8-b863-88638fc4f44f
# ╠═92e13255-b5e8-4127-afdc-978ec57f686f
# ╠═b5e3fdd5-c9cf-4c1b-8d1f-ffe60d663fee
# ╠═2d66b01a-5be9-452c-b9e3-dcd8a760db48
# ╟─f4408d42-71ab-43a2-94fd-e7f9df15fbb7
# ╠═18ae0915-f9c1-4564-b702-962e4ac897d0
# ╠═8a77505c-8be1-48e3-9c86-accd8368f571
# ╟─a426dc7d-06b9-499b-88aa-7cf8be1b99f4
# ╠═335ab9b1-e7be-46af-a6a6-3171adfda0d1
# ╠═3ca2874c-2596-4b5d-8edb-af231023feaf
# ╠═98c4315a-c235-4881-9ca6-95e2c8e49bcb
# ╟─ec517be9-2426-484f-86ba-4b339bb3db00
# ╠═3e3f14f7-51ad-42fe-95a0-d0eea4c8623b
# ╠═b7096208-eb1a-4b26-9337-a9f66a6f9682
# ╠═d08a483d-4823-4cff-9d68-89a29005e51a
# ╠═3135ae24-c5c3-41cb-ad78-f2eec786ee8e
