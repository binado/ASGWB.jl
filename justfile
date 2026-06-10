fmt:
    julia -e 'using JuliaFormatter; format(".")'

test:
    julia --project=ASGWB -e 'using Pkg; Pkg.test()'
    julia --project=ASGWBInference -e 'using Pkg; Pkg.test()'
    julia --project=CBCDistributions -e 'using Pkg; Pkg.test()'

pluto threads='"auto"':
    julia -e 'using Pluto; Pluto.run(threads={{threads}})'

resolve package="ASGWB":
    julia --project={{package}} -e 'using Pkg; Pkg.resolve()'

repl project=".":
    julia --project={{project}}

sync-notebook:
    jupytext 'notebooks/*.ipynb' --to jl:percent
