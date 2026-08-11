"""
    reconstruct_amplitude(rng, amplitude_mle, template_optimal_snr, template_merger_rate;
                          amplitude_fn, merger_rate_fn, prior, fiducial,
                          grid = quadrature_grid(prior))
        -> (; parameter, total_merger_rate, quadrature_effective_nodes)

Recover the marginalized physical parameter in post-processing from the three sufficient
statistics `astrosgwb_amplitude_marginalized_turing_model` publishes.

For every element of the (broadcast-compatible) inputs -- production passes the
`(draw, chain)` matrices straight off the saved chain -- this builds one
[`AmplitudeConditional`](@ref), draws one ``\\varphi`` from it, and returns

- `parameter` -- the reconstructed ``\\varphi`` (e.g. `H0`), *not* the amplitude;
- `total_merger_rate` -- the physical rate,
  `template_merger_rate * g_R(\\varphi) / g_R(\\varphi_\\mathrm{fid})`. The model
  deliberately publishes only `template_merger_rate`, the rate at the pinned fiducial
  amplitude, so no number that would be mistaken for the real merger rate is ever written
  at an unmarginalized ``\\varphi``;
- `quadrature_effective_nodes` -- [`effective_nodes`](@ref) per draw, the grid-adequacy
  diagnostic measured on the refined mesh the draws come from. Values below about 30 mean
  the conditional posterior is not resolved and the reconstructed draws are
  lattice-quantized; warn on the minimum. The marginalization the chain already did is far
  more forgiving -- see the module docstring in `amplitude.jl`.

There is no forward physics here -- no catalog, no `(nfreq, nsamples)` contraction -- so
the cost is O(length(grid)) per draw and this runs against a saved chain alone.

!!! warning "The conditional must be the one the chain integrated"

    `prior`, `fiducial`, and `amplitude_fn` must be **exactly** those
    `astrosgwb_amplitude_marginalized_turing_model` was given: they define the density the
    chain's `@addlogprob!` actually integrated. The `grid` is only quadrature accuracy --
    `quantile` refines a localized mesh from it and normalizes by `cdf[end]`, so any grid
    covering the prior support with enough nodes reconstructs the same draws. A silently
    different *definition* yields a wrong marginalized posterior with **no visible
    symptom**, because the sufficient statistics stay finite and plausible whatever
    conditional you pair them with.

`rng` should be seeded distinctly from the sampler -- these are fresh random draws, not a
deterministic function of the chain.
"""
function reconstruct_amplitude(
        rng::Random.AbstractRNG,
        amplitude_mle,
        template_optimal_snr,
        template_merger_rate;
        amplitude_fn,
        merger_rate_fn,
        prior::Distributions.ContinuousUnivariateDistribution,
        fiducial::Real,
        grid::AbstractVector{<:Real} = quadrature_grid(prior)
)
    merger_rate_fid = merger_rate_fn(fiducial)
    drawn = map(amplitude_mle, template_optimal_snr, template_merger_rate) do Â, ρ, rate
        conditional = AmplitudeConditional(Â, ρ; amplitude_fn, prior, fiducial, grid)
        φ = rand(rng, conditional)
        return (φ, rate * merger_rate_fn(φ) / merger_rate_fid, effective_nodes(conditional))
    end
    return (;
        parameter = map(first, drawn),
        total_merger_rate = map(t -> t[2], drawn),
        quadrature_effective_nodes = map(last, drawn)
    )
end
