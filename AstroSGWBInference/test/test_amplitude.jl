using Test
using Distributions: Uniform, Normal, ContinuousUnivariateDistribution,
                     logpdf, quantile, mean, std, insupport
using Random: Xoshiro
using AstroSGWBInference: AmplitudeConditional, quadrature_grid, log_normalizer,
                          effective_nodes, reconstruct_amplitude

# Independent of Trapezoid.trapz, so the tests below check the quadrature rather than
# restating it.
function _reference_trapezoid(f, x)
    y = f.(x)
    return sum(0.5 .* (y[1:(end - 1)] .+ y[2:end]) .* diff(collect(x)))
end

function _conditional(
        Â, ρ; prior = Uniform(0.5, 1.5), fiducial = 1.0, amplitude_fn = identity,
        kwargs...)
    AmplitudeConditional(
        Â, ρ; amplitude_fn, prior, fiducial, kwargs...)
end

@testset "quadrature_grid" begin
    # A Uniform's own support is tighter than ten standard deviations, so the grid
    # collapses onto its exact bounds -- the obvious grid, derived not special-cased.
    grid = quadrature_grid(Uniform(0.5, 1.5))
    @test first(grid) == 0.5
    @test last(grid) == 1.5
    @test length(grid) == 1024
    @test issorted(grid)

    # A Normal is unbounded, so `span_sigma` alone sets the span.
    normal_grid = quadrature_grid(Normal(3.0, 2.0); num_nodes = 11, span_sigma = 4.0)
    @test first(normal_grid) ≈ 3.0 - 8.0
    @test last(normal_grid) ≈ 3.0 + 8.0
    @test length(normal_grid) == 11

    @test_throws ArgumentError quadrature_grid(Uniform(0.5, 1.5); num_nodes = 1)
    @test_throws ArgumentError quadrature_grid(Uniform(0.5, 1.5); span_sigma = 0.0)
end

@testset "log_normalizer against a brute-force integral" begin
    c = _conditional(1.05, 30.0)
    # The unnormalized integrand, restated here from the module docstring's ℓ(φ) rather
    # than reused from the implementation.
    integrand(φ) = exp(logpdf(Uniform(0.5, 1.5), φ) -
                       0.5 * (30.0 * (φ / 1.0 - 1.05))^2)
    fine = range(0.5, 1.5; length = 200_001)
    # Loose on purpose: this compares a 1024-node trapezoid against a 200k-node one, so it
    # is bounded by the *grid's* quadrature error, not the rule's. It still pins the
    # integrand -- a wrong sign, a missing ½, or an unanchored amplitude is off by orders
    # of magnitude, not by 1e-6.
    @test log_normalizer(c) ≈ log(_reference_trapezoid(integrand, fine)) rtol = 1.0e-5
    # Exact against the same grid: this is the rule itself, with the max-shift removed.
    @test log_normalizer(c) ≈ log(_reference_trapezoid(integrand, c.grid)) rtol = 1.0e-12

    # A non-power-law scaling exercises the f(φ)/f(φ_fid) anchoring: the amplitude is 1 at
    # the fiducial by construction, never by trusting `amplitude_fn` to be normalized.
    inv_c = _conditional(0.9, 12.0; amplitude_fn = inv, fiducial = 1.2)
    inv_integrand(φ) = exp(logpdf(Uniform(0.5, 1.5), φ) -
                           0.5 * (12.0 * ((1 / φ) / (1 / 1.2) - 0.9))^2)
    @test log_normalizer(inv_c) ≈ log(_reference_trapezoid(inv_integrand, fine)) rtol = 1.0e-5
    @test log_normalizer(inv_c) ≈
          log(_reference_trapezoid(inv_integrand, inv_c.grid)) rtol = 1.0e-12
end

@testset "the max-shift survives a high-SNR conditional" begin
    # ρ² = 1e12 overflows nothing here only because the implementation squares
    # `ρ (A - Â)` rather than forming `ρ² (A - Â)²`, and the trapezoid is max-shifted.
    c = _conditional(1.0, 1.0e6)
    @test isfinite(log_normalizer(c))
    @test isfinite(logpdf(c, 1.0))
end

@testset "logpdf normalizes and respects the prior's support" begin
    c = _conditional(1.05, 30.0)
    @test _reference_trapezoid(φ -> exp(logpdf(c, φ)), c.grid) ≈ 1.0 rtol = 1.0e-6

    # The support is the *prior's*, and `logpdf` is analytic off the grid nodes.
    @test logpdf(c, 2.0) == -Inf
    @test logpdf(c, 0.4) == -Inf
    @test isfinite(logpdf(c, 1.0123456789))
end

@testset "quantile inverts the tabulated CDF" begin
    c = _conditional(1.05, 30.0)
    qs = 0.0:0.05:1.0
    xs = quantile.(Ref(c), qs)

    @test issorted(xs)
    # Draws are clipped to the refined mesh -- tighter than the prior-span grid, which is
    # itself slightly tighter than the declared support. Deliberate (module docstring).
    @test all(first(c.grid) .<= xs .<= last(c.grid))
    @test quantile(c, 0.0) <= xs[2]

    # The empirical CDF of a large sample must match the analytic density's.
    rng = Xoshiro(20260811)
    draws = [rand(rng, c) for _ in 1:50_000]
    for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        x = quantile(c, q)
        analytic = _reference_trapezoid(
            φ -> exp(logpdf(c, φ)), range(first(c.grid), x; length = 20_001))
        @test count(<=(x), draws) / length(draws) ≈ analytic atol = 0.01
        @test analytic ≈ q atol = 0.005
    end
end

@testset "the CDF-plateau guard keeps sharp conditionals finite" begin
    # ρ = 1e5 against a unit-width prior: the integrand underflows to exactly zero over
    # almost the whole grid, so the tabulated CDF is flat there. Unguarded, the
    # linear-in-CDF inversion divides 0/0 and every such draw is NaN.
    sharp = _conditional(1.0, 1.0e5)
    draws = rand.(Xoshiro.(1:500), Ref(sharp))
    @test all(isfinite, draws)
    @test all(first(sharp.grid) .<= draws .<= last(sharp.grid))
    # Anti-vacuity: the plateau really is there.
    @test count(iszero, exp.([logpdf(sharp, φ) for φ in sharp.grid])) > 900
end

@testset "effective_nodes flags an unresolved grid" begin
    # Same conditional, two grids. The default 1024-node grid resolves it comfortably
    # after refinement; a 16-node grid does not -- the fine mesh inherits the node count,
    # so a coarse mesh that is merely 16 nodes wide stays unresolved -- even though
    # `log_normalizer` returns a perfectly finite, plausible-looking number in both cases.
    well_resolved = _conditional(1.05, 30.0)
    @test effective_nodes(well_resolved) > 30

    coarse = _conditional(1.05, 30.0; num_nodes = 16)
    @test effective_nodes(coarse) < 30
    @test isfinite(log_normalizer(coarse))

    # And a conditional far too sharp for any reasonable node budget: ρ = 1e5 puts the
    # whole posterior inside a fraction of one coarse cell, so even the padded, refined
    # bracket spans ~3 coarse cells worth of nodes at default resolution.
    @test effective_nodes(_conditional(1.0, 1.0e5)) < 30
end

@testset "two-pass refinement resolves the production ρ = 400 conditional" begin
    # The case from QUADRATURE_NODES.md: on a prior-wide 1024-node inversion this
    # conditional inflated the 68% width by +9.3%, erred on the 5th percentile by 23% of
    # a coarse cell, and rated effective_nodes = 5.1 -- while log Z was already at machine
    # precision. The two-pass mesh must close all three at the same node budget.
    prior = Uniform(20.0, 140.0)
    fiducial = 67.66
    c = AmplitudeConditional(1.002, 400.0; amplitude_fn = inv, prior, fiducial)
    reference = AmplitudeConditional(
        1.002, 400.0; amplitude_fn = inv, prior, fiducial, num_nodes = 1_000_001)

    h = step(c.grid)
    for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        @test abs(quantile(c, q) - quantile(reference, q)) < 0.01h
    end

    width(conditional) = quantile(conditional, 0.84) - quantile(conditional, 0.16)
    @test width(c) ≈ width(reference) rtol = 1.0e-3

    # The diagnostic now reads the refined mesh, so it clears the threshold on the same
    # prior-span grid that used to report 5.1.
    @test effective_nodes(c) > 30
end

@testset "reconstruct_amplitude" begin
    prior = Uniform(0.5, 1.5)
    fiducial = 1.0
    # `g_R = φ²` is deliberately *not* `f = φ`: the reconstructed rate must scale by the
    # merger-rate piece alone, not by the full amplitude.
    merger_rate_fn(φ) = φ^2

    amplitude_mle = fill(1.05, 200, 4)
    template_optimal_snr = fill(30.0, 200, 4)
    template_merger_rate = fill(2.5, 200, 4)
    out = reconstruct_amplitude(
        Xoshiro(7), amplitude_mle, template_optimal_snr, template_merger_rate;
        amplitude_fn = identity, merger_rate_fn, prior, fiducial)

    @test size(out.parameter) == (200, 4)
    @test size(out.total_merger_rate) == (200, 4)
    @test size(out.quadrature_effective_nodes) == (200, 4)
    @test all(isfinite, out.parameter)

    # The rate scales as g_R(φ)/g_R(φ_fid), never as f(φ)/f(φ_fid).
    @test out.total_merger_rate ≈
          2.5 .* merger_rate_fn.(out.parameter) ./
          merger_rate_fn(fiducial)
    @test all(>(30), out.quadrature_effective_nodes)

    # The draws follow the conditional the statistics came from.
    reference = AmplitudeConditional(1.05, 30.0; amplitude_fn = identity, prior, fiducial)
    for q in (0.25, 0.5, 0.75)
        x = quantile(reference, q)
        @test count(<=(x), out.parameter) / length(out.parameter) ≈ q atol = 0.03
    end
end
