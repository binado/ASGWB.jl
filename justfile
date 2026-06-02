fmt:
    julia -e 'using JuliaFormatter; format(".")'

test:
    julia --project=ASGWB -e 'using Pkg; Pkg.test()'
    julia --project=ASGWBInference -e 'using Pkg; Pkg.test()'

pluto:
    julia --project=notebooks -e 'using Pkg; Pkg.instantiate(); using Pluto; Pluto.run(notebook="notebooks/mcmc_pluto.jl")'

resolve package="ASGWB":
    julia --project={{package}} -e 'using Pkg; Pkg.resolve()'

repl project=".":
    julia --project={{project}}

sync-notebook:
    jupytext 'notebooks/*.ipynb' --to jl:percent
