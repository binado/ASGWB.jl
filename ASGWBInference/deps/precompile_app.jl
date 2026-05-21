using ASGWB
using ASGWBInference

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PRECOMPILE_CACHE = joinpath(@__DIR__, "parity_precompile.h5")

include(joinpath(REPO_ROOT, "ASGWB", "test", "parity_test_cache.jl"))
write_parity_cache_h5(PRECOMPILE_CACHE, :posterior)

ENV["ASGWB_REPO_ROOT"] = REPO_ROOT
ENV["MCMC_CONFIG_FILEPATH"] = "config/run_inference_precompile.toml"

ASGWBInference.run_inference_from_env()
