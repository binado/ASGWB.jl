"""
Numerical marginalization of a multiplicative amplitude direction.

Under the per-frequency Gaussian likelihood of
[`astrosgwb_importance_turing_model`](@ref), one hyperparameter can enter the predicted
spectrum as a pure multiplicative factor,

``\\mu(\\varphi, \\theta) = A(\\varphi)\\, m(\\theta)``,

with ``m(\\theta)`` the *template* -- the spectrum at a fixed reference value
``\\varphi_\\mathrm{fid}`` of the marginalized parameter -- and

``A(\\varphi) = f(\\varphi) / f(\\varphi_\\mathrm{fid})``

the dimensionless amplitude relative to that template, for an arbitrary scaling ``f``.
Normalizing by ``f(\\varphi_\\mathrm{fid})`` here rather than trusting ``f`` to already
satisfy ``f(\\varphi_\\mathrm{fid}) = 1`` makes the anchoring structurally impossible to
get wrong, and is the correct construction for a non-power-law ``f``. The spectrum
factorizes into a total merger rate and a mean energy flux, so ``f = g_R \\cdot g_F``; see
`AstroSGWBImportanceModels.bns_amplitude_scalings` for the concrete `H0` and `R₀`
scalings.

With the σ-space inner product ``(x|y) = \\sum_i x_i y_i / \\sigma_i^2`` -- which is
exactly [`AstroSGWB.inner_product`](@ref), *not* a PSD-space contraction -- the amplitude
sufficient statistics are

``\\hat{A} = (d|m)/(m|m)``, ``\\rho = \\sqrt{(m|m)}``,

and completing the square in ``A`` gives

``-\\tfrac12 \\sum_i ((d_i - A m_i)/\\sigma_i)^2 = -R - \\tfrac12 \\rho^2 (A - \\hat A)^2``

with ``R = \\tfrac12 \\sum_i ((d_i - \\hat A m_i)/\\sigma_i)^2`` the best-fit residual.
This module marginalizes ``\\varphi`` numerically under the caller's actual prior
``\\pi(\\varphi)``, rather than requiring the prior to be stated on ``A`` itself. The
log-integrand is

``\\ell(\\varphi) = \\ln \\pi(\\varphi) - \\tfrac12 (\\rho (A(\\varphi) - \\hat A))^2``,

integrated by a max-shifted trapezoid rule on a fixed 1D grid. Squaring
``\\rho (A(\\varphi) - \\hat A)`` rather than forming ``\\rho^2 (A - \\hat A)^2`` avoids
overflowing ``\\rho^2`` at very high SNR, and is load-bearing there.

**The grid is a quadrature scheme, not the distribution.** The support is the *prior's*,
and [`Distributions.logpdf`](@ref) evaluates the analytic density at any ``\varphi``
without touching the grid. The one place the asymmetry shows is `rand`/`quantile`, which
invert a tabulated CDF and therefore return draws clipped to the mesh -- slightly tighter
than the declared support. That is deliberate: the grid must cover essentially all the
prior mass anyway (see [`quadrature_grid`](@ref)), or the normalizer is wrong for a
reason no amount of clipping would fix.

The two grid consumers have completely different error behaviour. [`log_normalizer`](@ref)
is a global sum whose trapezoid corrections are all endpoint differences, so with the
conditional posterior a bump deep inside a prior-span grid it is spectrally accurate --
~1024 nodes suffice even when the bump spans only a couple of cells. `rand`/`quantile` are
*local* readouts whose resolution is capped at the cell size, so they run a **two-pass**
scheme: bracket the posterior mass on the prior-span grid (cheap and forgiving -- the
max-shift pins the bracket to the peak's cell even when the integrand underflows
everywhere else), pad one coarse cell each side, then rebuild and invert the CDF on an
equally dense mesh spanning just that bracket. Even so, grid adequacy must be **checked**
with [`effective_nodes`](@ref), which measures the refined mesh that actually backs the
draw: a value below about 30 there means the conditional genuinely cannot be resolved at
that node count.

Unlike the Python original this distribution is **scalar**: one draw's statistics, not a
batch. Batching there is a JAX/`Predictive` requirement; here `reconstruct_amplitude`
simply maps over the `(draw, chain)` matrices, and a scalar distribution is what the
Distributions.jl interface expects.
"""

"""
    quadrature_grid(prior; num_nodes = 1024, span_sigma = 10.0) -> AbstractRange

A quadrature grid covering essentially all of `prior`'s mass.

Spans ± `span_sigma` prior standard deviations about the prior mean, clipped to the
prior's support. That reproduces the obvious grid for the two priors that matter in
practice: a `Uniform` collapses onto its exact `[low, high]` bounds (its own support is
tighter than ten standard deviations), and a `Normal` spans `μ ± span_sigma·σ`. At the
default `span_sigma = 10.0` the lost `Normal` tail mass is of order `1e-23`, a constant
offset identical for every posterior draw, so it does not perturb NUTS.

The grid **must** cover the prior support: the normalizing integral in
[`log_normalizer`](@ref) runs over exactly this grid, so narrowing it truncates the prior.

`Distributions.minimum`/`maximum`/`mean`/`std` do all the work, so this is generic over
any prior implementing them.
"""
function quadrature_grid(
        prior::Distributions.UnivariateDistribution;
        num_nodes::Int = 1024,
        span_sigma::Real = 10.0
)
    num_nodes > 1 || throw(ArgumentError("num_nodes must be > 1; got $num_nodes"))
    span_sigma > 0 || throw(ArgumentError("span_sigma must be > 0; got $span_sigma"))
    half_width = span_sigma * Distributions.std(prior)
    center = Distributions.mean(prior)
    lower = max(center - half_width, minimum(prior))
    upper = min(center + half_width, maximum(prior))
    isfinite(lower) && isfinite(upper) || throw(ArgumentError(
        "quadrature grid bounds are not finite ($lower, $upper); pass an explicit grid",
    ))
    return range(lower, upper; length = num_nodes)
end

"""
    AmplitudeConditional(amplitude_mle, template_optimal_snr;
                         amplitude_fn, prior, fiducial,
                         grid = quadrature_grid(prior; num_nodes, span_sigma))

Conditional posterior of the marginalized parameter given the amplitude statistics
``\\hat A`` and ``\\rho``:

``p(\\varphi \\mid d, \\theta) \\propto \\pi(\\varphi)
\\exp[-\\tfrac12 (\\rho (A(\\varphi) - \\hat A))^2]``,
``A(\\varphi) = f(\\varphi)/f(\\varphi_\\mathrm{fid})``.

It owns the **live** pieces it is defined by -- the prior, the scaling `amplitude_fn`, the
fiducial -- rather than a precomputed tabulation, so nothing can go stale. See the
module-level docstring in `amplitude.jl` for the grid-versus-support asymmetry.

The three consumers, all backed by the single `_log_density` implementation:

- [`log_normalizer`](@ref) -- ``\\ln Z``, which *is* the marginalization factor
  `astrosgwb_amplitude_marginalized_turing_model` adds to the log-likelihood at the MLE;
- `rand` / `Distributions.quantile` -- inverse-transform draws of ``\varphi`` for
  post-processing reconstruction on a two-pass refined mesh (see
  [`Distributions.quantile`](@ref)), clipped to that mesh;
- [`effective_nodes`](@ref) -- the grid-adequacy diagnostic.

The statistics arrive as `ForwardDiff.Dual`s inside the model body, so `amplitude_mle`,
`template_optimal_snr`, and `fiducial` are promoted to a common type rather than pinned
to `Float64`.
"""
struct AmplitudeConditional{
    T <: Real, F, D <: Distributions.ContinuousUnivariateDistribution,
    G <: AbstractVector{<:Real}} <:
       Distributions.ContinuousUnivariateDistribution
    amplitude_mle::T
    template_optimal_snr::T
    amplitude_fn::F
    prior::D
    fiducial::T
    grid::G
end

function AmplitudeConditional(
        amplitude_mle::Real,
        template_optimal_snr::Real;
        amplitude_fn,
        prior::Distributions.ContinuousUnivariateDistribution,
        fiducial::Real,
        num_nodes::Int = 1024,
        span_sigma::Real = 10.0,
        grid::AbstractVector{<:Real} = quadrature_grid(prior; num_nodes, span_sigma)
)
    Â, ρ, φ_fid = promote(amplitude_mle, template_optimal_snr, fiducial)
    return AmplitudeConditional(Â, ρ, amplitude_fn, prior, φ_fid, grid)
end

"""
    _log_density(c::AmplitudeConditional, φ) -> Real

Unnormalized ``\\ell(\\varphi)`` at an arbitrary ``\\varphi``.

The single implementation behind the normalizer, the density, and the inverse-CDF draw --
which is what keeps them from drifting apart.
"""
function _log_density(c::AmplitudeConditional, φ::Real)
    amplitude = c.amplitude_fn(φ) / c.amplitude_fn(c.fiducial)
    scaled_residual = c.template_optimal_snr * (amplitude - c.amplitude_mle)
    return Distributions.logpdf(c.prior, φ) - 0.5 * scaled_residual^2
end

"""
    _log_integrand(c::AmplitudeConditional) -> Vector

``\\ell`` evaluated on the quadrature grid. Recomputed rather than cached: the grid is
typically 1024 nodes of scalar arithmetic against an `(nfreq, nsamples)` contraction
upstream, so it does not register.
"""
_log_integrand(c::AmplitudeConditional) = _log_density.(c, c.grid)

"""
Tail mass fraction cut from each side when [`_fine_mesh`](@ref) localizes the refinement
mesh for `quantile`/`rand`/`effective_nodes`. `1e-6` is ≈ 4.75σ for a Gaussian bump; the
one-cell padding on top absorbs the coarse pass's only job, locating the peak.
"""
const _REFINE_EPSILON = 1e-6

"""
    _quantile_from_cdf(cdf, grid, q) -> Real

Inverse CDF by linear-in-CDF inversion: the answer sits in the one cell where `cdf`
crosses `q`, approximated as a straight line. `cdf` must start at 0 and be normalized.

`count(<(q), cdf)` is the number of nodes strictly below `q`; clamping to `[1, n-1]`
picks the bracketing cell `[i, i+1]` even for `q = 0` or `q = 1`. Deep in the tails the
shifted integrand underflows to 0, so the CDF has long flat plateaus; the division guard
lands those draws at `grid_lo` instead of a NaN from 0/0.
"""
function _quantile_from_cdf(cdf::AbstractVector, grid::AbstractVector, q::Real)
    n = length(grid)
    i = clamp(count(<(q), cdf), 1, n - 1)
    cdf_lo, cdf_hi = cdf[i], cdf[i + 1]
    grid_lo, grid_hi = grid[i], grid[i + 1]
    fraction = cdf_hi > cdf_lo ? (q - cdf_lo) / (cdf_hi - cdf_lo) : zero(q)
    return grid_lo + fraction * (grid_hi - grid_lo)
end

"""
    _fine_mesh(c::AmplitudeConditional) -> (grid, shifted_integrand)

The two-pass localized mesh backing `quantile`/`rand`/`effective_nodes`.

Inverse-CDF inversion is a *local* readout: its resolution is capped at the cell size, so
on a prior-span grid a conditional posterior of width ``\\sigma_\\varphi`` is quantized to
`h` -- at ρ ≈ 400 over `Uniform(20, 140)` that inflates the 68% width by ~9% at 1024
nodes (see `QUADRATURE_NODES.md`). The fix is a cheap localization pass followed by a
dense re-mesh:

1. Coarse CDF of the (max-shifted) integrand on `c.grid`.
2. Bracket the posterior mass between the `_REFINE_EPSILON` and `1 - _REFINE_EPSILON`
   quantiles, padded one coarse cell each side. Index-based padding makes no uniformity
   assumption on the grid, and the max-shift pins the bracket to the peak's cell even when
   the integrand underflows everywhere else -- so this survives σ/h < 1 on the coarse
   mesh, the regime where a non-shifted rule would lose the peak entirely.
3. Rebuild grid and integrand on an equally dense `range(lo, hi; length(c.grid))` mesh.
"""
function _fine_mesh(c::AmplitudeConditional)
    log_y = _log_integrand(c)
    shifted = exp.(log_y .- maximum(log_y))
    cdf = cumtrapz(c.grid, shifted)
    cdf ./= cdf[end]

    n = length(c.grid)
    i_lo = clamp(count(<(_REFINE_EPSILON), cdf), 1, n - 1)
    i_hi = clamp(count(<(1 - _REFINE_EPSILON), cdf), 1, n - 1)
    lo = c.grid[max(i_lo - 1, 1)]
    hi = c.grid[min(i_hi + 2, n)]

    fine = range(lo, hi; length = n)
    fine_log_y = _log_density.(c, fine)
    return fine, exp.(fine_log_y .- maximum(fine_log_y))
end

"""
    log_normalizer(c::AmplitudeConditional) -> Real

``\\ln Z`` of the conditional -- **the marginalization factor itself**.

A max-shifted ``\\ln \\int \\exp(\\ell)\\, d\\varphi`` over `c.grid`.
`astrosgwb_amplitude_marginalized_turing_model` adds exactly this to the log-likelihood at
the MLE amplitude: the factor *is* the normalizing constant of the conditional that
[`reconstruct_amplitude`](@ref) later draws from.
"""
function log_normalizer(c::AmplitudeConditional)
    log_y = _log_integrand(c)
    log_y_max = maximum(log_y)
    return log_y_max + log(trapz(c.grid, exp.(log_y .- log_y_max)))
end

"""
    effective_nodes(c::AmplitudeConditional) -> Real

Grid-adequacy diagnostic for the **reconstruction draws**: how many nodes of the refined
mesh [`_fine_mesh`](@ref) actually carry the conditional posterior.

Reuses [`normalized_ess`](@ref) on the shifted fine integrand -- the same Kish
effective-sample-size construction used for the importance weights -- rescaled by the node
count so the result reads as a node count rather than a fraction. This should be
comfortably above about **30**. It gates `rand`/`quantile` alone: [`log_normalizer`](@ref)
is spectrally accurate on the coarse grid far below that threshold, so a passing value
here is *not* what makes the marginal likelihood right, and a failing one means the
reconstructed draws -- not the chain -- are lattice-quantized.
"""
function effective_nodes(c::AmplitudeConditional)
    _, fine_shifted = _fine_mesh(c)
    return normalized_ess(fine_shifted) * length(c.grid)
end

Base.minimum(c::AmplitudeConditional) = minimum(c.prior)
Base.maximum(c::AmplitudeConditional) = maximum(c.prior)

"""
    Distributions.logpdf(c::AmplitudeConditional, φ) -> Real

Exact log density, evaluated **analytically off the grid**. The normalizing constant is
still the trapezoid integral over the grid, so this integrates to 1 only up to quadrature
error.

`Distributions.Uniform`'s own `logpdf` already returns `-Inf` off support, but the explicit
`insupport` guard is what keeps this consistent with `minimum`/`maximum` for any prior.
"""
function Distributions.logpdf(c::AmplitudeConditional, φ::Real)
    log_density = _log_density(c, φ) - log_normalizer(c)
    return Distributions.insupport(c.prior, φ) ? log_density : oftype(log_density, -Inf)
end

"""
    Distributions.quantile(c::AmplitudeConditional, q) -> Real

Inverse CDF of the conditional by two-pass inversion, so draws follow precisely the
density that was marginalized -- a piecewise-*constant* approximation of it.

The prior-span grid resolves the *integral* spectrally but a *location* only to the cell
size, so the CDF is first bracketed coarsely on `c.grid` and then rebuilt on the refined
mesh from [`_fine_mesh`](@ref), where the actual inversion happens. Normalizing by
`cdf[end]` makes the unknown normalizer cancel out of the draw, which is why the mesh may
differ from the one [`log_normalizer`](@ref) integrated. Returns ``\varphi`` clipped to
the refined mesh.
"""
function Distributions.quantile(c::AmplitudeConditional, q::Real)
    fine, fine_shifted = _fine_mesh(c)
    cdf = cumtrapz(fine, fine_shifted)
    cdf ./= cdf[end]
    return _quantile_from_cdf(cdf, fine, q)
end

"""
    rand(rng, c::AmplitudeConditional) -> Real

Inverse-transform draw: one uniform through [`Distributions.quantile`](@ref).
"""
function Base.rand(rng::Random.AbstractRNG, c::AmplitudeConditional)
    return Distributions.quantile(c, rand(rng))
end
