using Distributions
using InteractiveUtils

using ASGWB

# Synthetic FullBNSSamples for a type-stability smoke test; requires a real
# RedshiftBundle for `intrinsic_prior(FullBNS(), bundle)`, so construct one via
# `build_redshift_grid_bundle` from representative hyperparameters.

theta = HyperParameters(;
    H0=67.0,
    Omega_m=0.315,
    gamma=2.7,
    kappa=3.0,
    z_peak=2.5,
)
spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
bundle = build_redshift_grid_bundle(theta, spec)
prior = intrinsic_prior(FullBNS(), bundle)

n = 10
samples = (
    mass=stack_source_masses(rand(n), rand(n)),
    redshift=rand(n),
    chi_1=rand(n),
    chi_2=rand(n),
    lambda_1=rand(n) .* 100,
    lambda_2=rand(n) .* 100,
)

@code_warntype intrinsic_log_prob_samples(prior, samples)
