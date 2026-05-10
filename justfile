fmt:
    julia -e 'using JuliaFormatter; format(".")'

test:
    julia --project=. -e 'using Pkg; Pkg.test()'

resolve project=".":
    julia --project={{project}} -e 'using Pkg; Pkg.resolve()'

repl project=".":
    julia --project={{project}}

sync-notebook:
    jupytext 'notebooks/*.ipynb' --to jl:percent

compile:
    julia --project=ASGWBInference ASGWBInference/deps/build.jl app

