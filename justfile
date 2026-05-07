fmt:
    julia -e 'using JuliaFormatter; format(".")'

test:
    julia --project=. -e 'using Pkg; Pkg.test()'

resolve:
    julia --project=. -e 'using Pkg; Pkg.resolve()'

notebook-dir := "notebooks"

# Launch a Pluto notebook: just pluto inference | just pluto postprocessing
pluto notebook="inference":
    #!/usr/bin/env bash
    case "{{notebook}}" in
        inference)       file="mcmc.jl" ;;
        postprocessing)  file="plots.jl" ;;
        *)
            echo "error: unknown notebook '{{notebook}}'. Use 'inference' or 'postprocessing'."
            exit 1 ;;
    esac
    export NOTEBOOK_FILE="{{notebook-dir}}/${file}"
    julia -tauto --project={{notebook-dir}} -e 'using Pkg; Pkg.instantiate(); using Pluto; Pluto.run(notebook=ENV["NOTEBOOK_FILE"])'
