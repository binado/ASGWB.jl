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
and [`Distributions.logpdf`](@ref) evaluates the analytic density at any ``\\varphi``
without touching the grid. The one place the asymmetry shows is `rand`/`quantile`, which
invert a CDF tabulated on the grid and therefore return draws clipped to
`[grid[1], grid[end]]` -- slightly tighter than the declared support. That is deliberate:
the grid must cover essentially all the prior mass anyway (see [`quadrature_grid`](@ref)),
or the normalizer is wrong for a reason no amount of clipping would fix.

"Exact up to quadrature error" only holds if the grid resolves the conditional posterior,
whose width in ``\\varphi`` is ``\\sigma_A / |A'(\\varphi)|``. No quadrature rule rescues a
Gaussian bump spanning three nodes, so grid adequacy must be **checked** with
[`effective_nodes`](@ref), not assumed.

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
- `rand` / `Distributions.quantile` -- inverse-transform draws of ``\\varphi`` for
  post-processing reconstruction, clipped to the grid;
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
_log_integrand(c::AmplitudeConditional) = [_log_density(c, φ) for φ in c.grid]

"""
    _trapezoid(y, x) -> Real

Trapezoid quadrature of `y` against abscissa `x`.
"""
function _trapezoid(y::AbstractVector, x::AbstractVector)
    total = zero(eltype(y)) * zero(eltype(x))
    @inbounds for i in firstindex(y):(lastindex(y) - 1)
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    end
    return total
end

"""
    _cumulative_trapezoid(y, x) -> Vector

Cumulative trapezoid integral of `y` against `x`, starting at 0 and the same length as `y`.
"""
function _cumulative_trapezoid(y::AbstractVector, x::AbstractVector)
    out = similar(y, typeof(zero(eltype(y)) * zero(eltype(x))), length(y))
    out[begin] = 0
    @inbounds for i in firstindex(y):(lastindex(y) - 1)
        out[i + 1] = out[i] + 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    end
    return out
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
    return log_y_max + log(_trapezoid(exp.(log_y .- log_y_max), c.grid))
end

"""
    effective_nodes(c::AmplitudeConditional) -> Real

Grid-adequacy diagnostic: how many nodes actually carry the conditional posterior.

Reuses [`normalized_ess`](@ref) on the shifted integrand -- the same Kish
effective-sample-size construction used for the importance weights -- rescaled by the node
count so the result reads as a node count rather than a fraction. A conditional posterior
spanning only a handful of nodes reports a small value here even though the assembled log
evidence looks finite and plausible; this should be comfortably above about **30**.
"""
function effective_nodes(c::AmplitudeConditional)
    log_y = _log_integrand(c)
    return normalized_ess(exp.(log_y .- maximum(log_y))) * length(c.grid)
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

Inverse CDF by linear-in-CDF inversion of the cumulative trapezoid of the same integrand
[`log_normalizer`](@ref) integrates, so draws follow precisely the density that was
marginalized -- a piecewise-*constant* approximation of it, which agrees with `logpdf` at
node resolution and differs sub-cell. Returns ``\\varphi`` clipped to
`[grid[1], grid[end]]`.
"""
function Distributions.quantile(c::AmplitudeConditional, q::Real)
    log_y = _log_integrand(c)
    shifted = exp.(log_y .- maximum(log_y))
    cdf = _cumulative_trapezoid(shifted, c.grid)
    cdf ./= cdf[end]

    n = length(c.grid)
    # `count(<(q), cdf)` is the number of nodes strictly below `q`; clamping to
    # `[1, n-1]` picks the bracketing cell `[i, i+1]` even for q = 0 or q = 1.
    i = clamp(count(<(q), cdf), 1, n - 1)
    cdf_lo, cdf_hi = cdf[i], cdf[i + 1]
    grid_lo, grid_hi = c.grid[i], c.grid[i + 1]

    # Deep in the tails `shifted` underflows to 0, so the CDF has long flat plateaus.
    # Guard the division: those draws land at `grid_lo` instead of a NaN from 0/0.
    fraction = cdf_hi > cdf_lo ? (q - cdf_lo) / (cdf_hi - cdf_lo) : zero(q)
    return grid_lo + fraction * (grid_hi - grid_lo)
end

"""
    rand(rng, c::AmplitudeConditional) -> Real

Inverse-transform draw: one uniform through [`Distributions.quantile`](@ref).
"""
function Base.rand(rng::Random.AbstractRNG, c::AmplitudeConditional)
    return Distributions.quantile(c, rand(rng))
end
