fmt:
    julia -e 'using JuliaFormatter; format(".")'


notebook-dir := "notebooks"
pluto:
    julia --project={{notebook-dir}} -e 'using Pkg; using Pluto; Pkg.instantiate(); Pluto.run()'
