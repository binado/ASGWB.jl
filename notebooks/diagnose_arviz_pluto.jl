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
    using MCMCChains
    using StatsPlots
    using NCDatasets
    using Statistics
end

# ╔═╡ 55bb7b84-631d-4c9a-9295-3ef239427d75
md"""
# ArviZ NetCDF diagnostics

Load a saved ArviZ `InferenceData` NetCDF file, convert to an `MCMCChains.Chains`
object, and inspect MCMC diagnostics without running a sampler. Edit only
`netcdf_path` when switching chain files.
"""

# ╔═╡ c66dee78-6a7e-4891-b90c-85959e2638b7
netcdf_path = joinpath(@__DIR__, "..", "chains-H0-γ-κ-zpeak.nc")

# ╔═╡ ef3f89c0-e204-4141-a985-26649d598d9e
begin
    isfile(netcdf_path) ||
        throw(ArgumentError("NetCDF file not found: $(repr(netcdf_path))"))
    idata = from_netcdf(netcdf_path)
    chain = to_mcmcchains(idata; group = "posterior")
end

# ╔═╡ e126abe3-591b-4143-a5bf-2af3390136c5
begin
    chain_params = names(chain, :parameters)
end

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
            max = maximum(values),
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
                   ArviZ.bfmi(variable_array(stats, :energy); dims = 1) : missing,
        )
    end

    nothing
end

# ╔═╡ a8dc8fc0-486a-4395-852b-857fb12e37d6
md"""
## Data
"""

# ╔═╡ 704aecb0-0157-4dae-b9b4-471489be1758
md"""
**InferenceData groups:** $(propertynames(idata))
"""

# ╔═╡ 1e617dd3-8749-44cc-a3a5-25f49a5e4023
chain_params

# ╔═╡ abf9ab52-3622-4106-b465-b6542b090040
md"""
## Numeric diagnostics
"""

# ╔═╡ f08e0893-fea6-4e86-9c7a-0287d9375e71
describe(chain)

# ╔═╡ 7b0a9e36-1df6-4eae-b740-32b5b40f342b
MCMCChains.ess(chain)

# ╔═╡ 2d66b01a-5be9-452c-b9e3-dcd8a760db48
sample_stats_diagnostics(idata)

# ╔═╡ f4408d42-71ab-43a2-94fd-e7f9df15fbb7
md"""
## Trace and autocorrelation plots
"""

# ╔═╡ 18ae0915-f9c1-4564-b702-962e4ac897d0
traceplot(chain)

# ╔═╡ 622dc36b-a0f2-482e-b562-70ee63d5904a
autocorplot(chain)

# ╔═╡ ec517be9-2426-484f-86ba-4b339bb3db00
md"""
## Posterior distributions
"""

# ╔═╡ d08a483d-4823-4cff-9d68-89a29005e51a
begin
    if length(chain_params) >= 2
        MCMCChains.corner(chain)
    else
        StatsPlots.density(chain)
    end
end

# ╔═╡ Cell order:
# ╠═3a065958-b6f1-4855-ad59-803892b592de
# ╟─55bb7b84-631d-4c9a-9295-3ef239427d75
# ╠═c66dee78-6a7e-4891-b90c-85959e2638b7
# ╠═ef3f89c0-e204-4141-a985-26649d598d9e
# ╠═e126abe3-591b-4143-a5bf-2af3390136c5
# ╠═7871eec2-3894-4ae3-981d-0c0a22cfb5fe
# ╟─a8dc8fc0-486a-4395-852b-857fb12e37d6
# ╠═704aecb0-0157-4dae-b9b4-471489be1758
# ╠═1e617dd3-8749-44cc-a3a5-25f49a5e4023
# ╟─abf9ab52-3622-4106-b465-b6542b090040
# ╠═f08e0893-fea6-4e86-9c7a-0287d9375e71
# ╠═7b0a9e36-1df6-4eae-b740-32b5b40f342b
# ╠═2d66b01a-5be9-452c-b9e3-dcd8a760db48
# ╟─f4408d42-71ab-43a2-94fd-e7f9df15fbb7
# ╠═18ae0915-f9c1-4564-b702-962e4ac897d0
# ╠═622dc36b-a0f2-482e-b562-70ee63d5904a
# ╟─ec517be9-2426-484f-86ba-4b339bb3db00
# ╠═d08a483d-4823-4cff-9d68-89a29005e51a
