fmt:
    julia -e 'using JuliaFormatter; format(".")'

test:
    julia --project=. -e 'using Pkg; Pkg.test()'

notebook-dir := "notebooks"
pluto:
    julia --project={{notebook-dir}} -e 'using Pkg; using Pluto; Pkg.instantiate(); Pluto.run()'
