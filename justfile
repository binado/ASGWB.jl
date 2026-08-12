fmt:
    julia -e 'using JuliaFormatter; format(".")'

test:
    julia --project=Trapezoid -e 'using Pkg; Pkg.test()'
    julia --project=BackgroundCosmology -e 'using Pkg; Pkg.test()'
    julia --project=GWBackground -e 'using Pkg; Pkg.test()'
    julia --project=GWBackgroundInference -e 'using Pkg; Pkg.test()'
    julia --project=GWBackgroundImportanceModels -e 'using Pkg; Pkg.test()'
    julia --project=GWDistributions -e 'using Pkg; Pkg.test()'

pluto threads='"auto"':
    julia -e 'using Pluto; Pluto.run(threads={{threads}})'

run-mcmc config="config/mcmc/example.toml" threads="auto":
    julia --project=scripts/run -e 'using Pkg; Pkg.instantiate()'
    julia --project=scripts/run -t {{threads}} scripts/run_mcmc.jl {{config}}

setup-run:
    julia --project=scripts/run -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

submit-mcmc config="config/mcmc/example.toml":
    mkdir -p logs
    sbatch scripts/submit_mcmc_single.sbatch {{config}}

submit-mcmc-array config_dir="config/mcmc/sweep" max_parallel="8":
    scripts/submit_mcmc_array.sh {{config_dir}} {{max_parallel}}

resolve package="GWBackground":
    julia --project={{package}} -e 'using Pkg; Pkg.resolve()'

repl project=".":
    julia --project={{project}}

sync-notebook:
    jupytext 'notebooks/*.ipynb' --to jl:percent
