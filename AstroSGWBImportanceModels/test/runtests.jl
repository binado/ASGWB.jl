using Test
using AstroSGWB
using AstroSGWBImportanceModels
using AstroSGWBInference
using AstroSGWBCosmology
using DataInterpolations: LinearInterpolation
using Distributions: Uniform
using ForwardDiff
using Turing

# S7: `R₀` (Gpc⁻³ yr⁻¹) is a live hyperparameter read as `Λ.R₀`, not a frozen struct
# field. It sits in both points at the same value the old `local_merger_rate` keyword
# carried, which is why the frozen `rate` fixture below does not move.
const FIDUCIALS = (
    H0 = 67.0,
    Ωm = 0.315,
    Ξ₀ = 1.0,
    Ξₙ = 0.0,
    γ = 2.7,
    κ = 3.0,
    zpeak = 2.5,
    R₀ = 161.0
)

const TARGET = (
    H0 = 70.0,
    Ωm = 0.3,
    Ξ₀ = 1.1,
    Ξₙ = 0.2,
    γ = 2.9,
    κ = 3.1,
    zpeak = 2.2,
    R₀ = 161.0
)

const SAMPLES = (
    redshift = [0.1, 0.2],
    luminosity_distance = [430.0, 880.0]
)

function prepared(samples = SAMPLES; C = LambdaCDM, P = ModifiedPropagation,
        fiducials = FIDUCIALS, z_grid = DEFAULT_Z_GRID)
    return prepare_bns_madau_dickinson_model(samples, fiducials, C, P; z_grid)
end

@testset "the prepared model is the contract callable" begin
    # S1: the whole model contract is `merger_rate_and_log_weights_fn(Λ, samples) -> (rate, log_weights)`.
    # No abstract supertype, no generic function to add methods to -- so this package
    # imports nothing from `AstroSGWBInference` and a plain closure would serve equally.
    model = prepared()
    @test !isempty(methods(model))
    rate, log_weights = model(TARGET, SAMPLES)
    @test rate isa Real
    @test length(log_weights) == length(SAMPLES.redshift)
end

@testset "preparation caches and fixed-fixture parity" begin
    model = prepared()
    @test model isa BNSMadauDickinsonImportanceModel{LambdaCDM, ModifiedPropagation}
    @test model.z_grid isa Vector{Float64}
    @test model.proposal_log_pdf isa Vector{Float64}
    @test length(model.z_grid) == length(DEFAULT_Z_GRID)
    @test length(model.proposal_log_pdf) == length(SAMPLES.redshift)
    @test all(isfinite, model.proposal_log_pdf)
    # All three refrozen when `DEFAULT_Z_GRID` moved from [1e-3, 20] to [0, 20] to match
    # `astrogwb.cosmology.distance_and_volume_grid`, which requires a grid starting at 0.
    # Unlike the S5/S6 refreeze, this one moves `proposal_log_pdf` and `rate` as well —
    # the grid *is* the integration domain, so restoring the missing first cell changes
    # both the density normalization (+0.32%) and `∫dN/dz` (+0.169%). Measured shift in
    # `log_weights`: -2.09e-2 at z = 0.1, -1.05e-2 at z = 0.2. Dropping the underflow
    # floor in the same commit contributes ~1e-15 absolute, i.e. nothing.
    @test model.proposal_log_pdf ≈ [-6.2539635957301094, -4.9113292473890375]

    rate, log_weights = model(TARGET, SAMPLES)
    @test rate ≈ 0.031168377918986516 rtol = 1.0e-13
    @test log_weights ≈ [-0.10995559838341759, -0.16653907807566956] rtol = 1.0e-12
    @test size(log_weights) == size(SAMPLES.redshift)
    @test all(isfinite, log_weights)

    @test_throws DimensionMismatch model(TARGET, (
        redshift = [0.1], luminosity_distance = [430.0]))
    # Decision B: the interpolator clamps, but a proposal sample off the integration grid
    # is a setup error and must be loud at prepare time.
    @test_throws ArgumentError prepared((
        redshift = [0.1, 25.0], luminosity_distance = [430.0, 880.0]))
    @test_throws ArgumentError prepared(; z_grid = [0.0])
    @test_throws ArgumentError prepared(; z_grid = [0.0, 1.0, 0.5])
end

@testset "prepare and hot path share one kernel" begin
    model = prepared()
    # `==`, not `≈`: `_bns_grid_terms` is literally the function `prepare` called, so at
    # Λ == fiducials the target density is bit-identical to the cached proposal density.
    # If these ever diverge, every posterior silently acquires a per-sample offset.
    @test AstroSGWBImportanceModels._bns_grid_terms(
        LambdaCDM, FIDUCIALS, model.z_grid, SAMPLES.redshift).log_p ==
          model.proposal_log_pdf

    # The full weight expression vanishes when `d_L_fid` is built through the *same* grid
    # path the hot path uses, so nothing is left but exact cancellation.
    z = SAMPLES.redshift
    d_l_grid = LinearInterpolation(
        distance_and_volume_grid(cosmology(LambdaCDM, FIDUCIALS),
            model.z_grid).luminosity_distance,
        model.z_grid
    )(z)
    grid_samples = (redshift = z, luminosity_distance = d_l_grid)
    _, w = model(FIDUCIALS, grid_samples)
    @test maximum(abs, w) < 1e-14
end

@testset "DEFAULT_Z_GRID starts at zero" begin
    # Comoving distance is accumulated along the grid assuming `d_c(grid[1]) = 0`:
    # `distance_and_volume_grid` documents the grid must start at zero but does not
    # check it (grid validation is caller-owned), so the production grid's zero lower
    # bound is asserted here instead.
    @test first(DEFAULT_Z_GRID) == 0.0
    @test last(DEFAULT_Z_GRID) == 20.0
    @test length(DEFAULT_Z_GRID) == 256
end

@testset "DataInterpolations linear sample evaluation" begin
    grid = [0.0, 0.5, 1.0, 2.0]
    values = [0.0, 1.0, 4.0, 8.0]
    points = [0.0, 0.25, 0.9, 2.0]
    expected = map(points) do z
        i = clamp(searchsortedlast(grid, z), 1, length(grid) - 1)
        values[i] + (values[i + 1] - values[i]) * (z - grid[i]) /
                    (grid[i + 1] - grid[i])
    end
    actual = AstroSGWBImportanceModels._linear_interpolate(values, grid, points)
    @test actual ≈ expected rtol = 1e-15

    empty = AstroSGWBImportanceModels._linear_interpolate(values, grid, Float64[])
    @test isempty(empty)
    @test eltype(empty) === Float64
end

@testset "S11 fiducial GW-distance reference" begin
    fid_gr = merge(FIDUCIALS, (Ξ₀ = 1.0, Ξₙ = 0.0))
    fid_mod = merge(FIDUCIALS, (Ξ₀ = 1.4, Ξₙ = 0.7))
    prop_fid = propagation(ModifiedPropagation, fid_mod)
    z = SAMPLES.redshift
    polarization_power = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]

    m_gr, m_mod = prepared(; fiducials = fid_gr), prepared(; fiducials = fid_mod)
    @test m_gr.log_Ξ_fid == zeros(length(z))
    # Isolates the change to log_Ξ_fid: the proposal density is propagation-independent.
    @test m_mod.proposal_log_pdf == m_gr.proposal_log_pdf
    _, w_gr = m_gr(TARGET, SAMPLES)
    _, w_mod = m_mod(TARGET, SAMPLES)
    @test w_mod ≈ w_gr .+ 2 .* log.(gw_em_distance_ratio.(z, Ref(prop_fid)))

    # The load-bearing invariant, stated on the physical contraction:
    # corrected polarization_power + Ξ_fid weights == uncorrected polarization_power + no-Ξ_fid weights.
    corrected = apply_gw_distance_correction(polarization_power, z, prop_fid)
    @test corrected ≉ polarization_power                                   # anti-vacuity guard
    @test corrected * exp.(w_mod) ≈ polarization_power * exp.(w_gr) rtol = 1e-13

    @test apply_gw_distance_correction!(polarization_power, z, GR()) === polarization_power
    @test apply_gw_distance_correction(polarization_power, z,
        propagation(ModifiedPropagation, fid_gr)) ≈ polarization_power
end

@testset "ForwardDiff empty and one-sample evaluations" begin
    dual(x) = ForwardDiff.Dual{Nothing}(x, one(x))
    Λ_dual = NamedTuple{keys(FIDUCIALS)}(map(dual, values(FIDUCIALS)))
    empty_samples = (redshift = Float64[], luminosity_distance = Float64[])
    one_sample = (redshift = [0.1], luminosity_distance = [500.0])

    empty_model = prepared(empty_samples)
    one_model = prepared(one_sample)
    empty_rate,
    empty_weights = empty_model(Λ_dual, empty_samples)
    one_rate, one_weights = one_model(Λ_dual, one_sample)

    @test isfinite(empty_rate)
    @test isfinite(one_rate)
    @test isempty(empty_weights)
    @test eltype(empty_weights) <: ForwardDiff.Dual
    @test length(one_weights) == 1
    @test all(isfinite, one_weights)
    @test eltype(one_weights) <: ForwardDiff.Dual
end

@testset "mixed-eltype Λ (partially sampled hyperparameters)" begin
    # A run that samples a subset leaves the rest `Float64` while the free ones become
    # `Dual`, so `propagation(P, Λ)` sees one `Dual` and one `Float64`. Before the
    # promoting `ModifiedPropagation` constructor this was a `MethodError`, and it is the
    # exact shape DynamicPPL conditioning produces on every gradient evaluation: free
    # coordinates are `Dual`, pinned ones `Float64`.
    model = prepared()
    dΞ₀ = ForwardDiff.derivative(1.1) do Ξ₀
        _, w = model(merge(TARGET, (; Ξ₀)), SAMPLES)
        sum(w)
    end
    @test isfinite(dΞ₀)
    @test !iszero(dΞ₀)

    # Same for a cosmology parameter, where only `Λ.H0` is dual.
    dH0 = ForwardDiff.derivative(70.0) do H0
        rate, _ = model(merge(TARGET, (; H0)), SAMPLES)
        rate
    end
    @test isfinite(dH0)
    @test !iszero(dH0)
end

@testset "concrete adapter integrates with Turing" begin
    model = prepared()
    polarization_power = Float64[1.0 1.5; 2.0 2.5]
    frequencies = [20.0, 40.0]
    eff_psd = [1.0, 1.0]
    observation_time = 1.0
    prior = (
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.0, 3.0),
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0),
        R₀ = Uniform(10.0, 1000.0)
    )
    # The prior declares every hyperparameter the model reads. `R₀` is pinned by
    # conditioning rather than sampled -- the production default; dropping the
    # conditioning samples it, with no change anywhere else. `observed` is synthesized
    # at the fiducials since there is no external spectrum to fit.
    observed = forward_model(
        model, polarization_power, SAMPLES, FIDUCIALS).spectral_density
    unconditioned = astrosgwb_importance_turing_model(
        model, polarization_power, SAMPLES, bns_hyperprior(prior), observed, frequencies,
        eff_psd, observation_time, AnalyticInclination())
    turing_model = unconditioned | (; R₀ = FIDUCIALS.R₀)

    Λ_sampled = Base.structdiff(FIDUCIALS, (; R₀ = nothing,))
    @test Set(Symbol.(keys(Turing.DynamicPPL.VarInfo(turing_model)))) ==
          Set(keys(Λ_sampled))
    @test isfinite(Turing.logjoint(turing_model, Λ_sampled))

    # And the opt-in: the unconditioned model samples every name in the prior.
    @test isfinite(Turing.logjoint(unconditioned, FIDUCIALS))
    @test Set(Symbol.(keys(Turing.DynamicPPL.VarInfo(unconditioned)))) ==
          Set(keys(FIDUCIALS))
end

# --------------------------------------------------------------------------
# Amplitude scalings
# --------------------------------------------------------------------------

@testset "amplitude scalings dispatch" begin
    @test AMPLITUDE_PARAMETERS == (:H0, :R₀)

    h0 = bns_amplitude_scalings(:H0)
    @test h0.amplitude_fn === amplitude_H0
    @test h0.merger_rate_fn === merger_rate_amplitude_H0
    r0 = bns_amplitude_scalings(:R₀)
    @test r0.amplitude_fn === amplitude_R₀
    @test r0.merger_rate_fn === merger_rate_amplitude_R₀

    # `f = g_R · g_F`: H0 has g_F = φ², R₀ has g_F = 1.
    @test amplitude_H0(70.0) ≈ merger_rate_amplitude_H0(70.0) * 70.0^2
    @test amplitude_R₀(161.0) == merger_rate_amplitude_R₀(161.0)

    # A parameter that is not strictly multiplicative must not silently marginalize.
    @test_throws ArgumentError bns_amplitude_scalings(:Ωm)
    @test_throws ArgumentError bns_amplitude_scalings(:zpeak)
end

@testset "the amplitude parameters are exactly multiplicative" begin
    # This is the assumption the entire marginalization rests on, and it has no Python
    # counterpart: `astrosgwb_amplitude_marginalized_turing_model` integrates
    # `A(φ) = f(φ)/f(φ_fid)` out of the likelihood analytically, which is only correct if
    # the *real* forward model factorizes as `μ(φ, θ) = A(φ) m(θ)`. Assert it against
    # `forward_model` itself rather than re-deriving the scalings.
    polarization_power = Float64[1.0 1.5; 2.0 2.5]
    # Production families: neither the dark-energy equation of state nor modified
    # propagation may spoil the factorization, so both are exercised away from their GR /
    # ΛCDM values.
    for (C, P, base) in (
        (LambdaCDM, ModifiedPropagation, FIDUCIALS),
        (W0CDM, ModifiedPropagation, merge(FIDUCIALS, (w0 = -0.8, Ξ₀ = 1.4, Ξₙ = 0.7)))
    )
        model = prepared(; C, P, fiducials = base)
        for (name, amplitude_fn) in ((:H0, amplitude_H0), (:R₀, amplitude_R₀))
            fid = base[name]
            template = forward_model(model, polarization_power, SAMPLES,
                merge(TARGET, base, NamedTuple{(name,)}((fid,)))).spectral_density
            for φ in (0.6 * fid, 0.85 * fid, fid, 1.3 * fid, 1.8 * fid)
                Λ = merge(TARGET, base, NamedTuple{(name,)}((φ,)))
                scaled = forward_model(
                    model, polarization_power, SAMPLES, Λ).spectral_density
                @test scaled ≈ (amplitude_fn(φ) / amplitude_fn(fid)) .* template rtol = 1.0e-10
            end
        end

        # Anti-vacuity: the spectrum really does move with these parameters, so the
        # assertions above are not comparing a constant against itself.
        for name in AMPLITUDE_PARAMETERS
            fid = base[name]
            @test !isapprox(
                forward_model(model, polarization_power, SAMPLES,
                    merge(TARGET, base, NamedTuple{(name,)}((fid,)))).spectral_density,
                forward_model(model, polarization_power, SAMPLES,
                    merge(TARGET, base,
                        NamedTuple{(name,)}((1.8 * fid,)))).spectral_density)
        end
    end
end

@testset "the BNS adapter marginalizes against the general likelihood" begin
    # End-to-end on the real adapter: the marginalized model's log density must equal the
    # numerically integrated general one. Same identity as the inference package's
    # equivalence test, but here the multiplicative structure comes from the physics
    # rather than from a three-line fixture.
    model = prepared()
    polarization_power = Float64[1.0 1.5; 2.0 2.5]
    frequencies = [20.0, 40.0]
    observation_time = 1.0

    amplitude_prior = Uniform(100.0, 250.0)
    grid = quadrature_grid(amplitude_prior; num_nodes = 4096)
    # The prior declares *every* hyperparameter name, sampled or not; `fixed` pins the
    # rest by conditioning, leaving `zpeak` as the only shape latent.
    full_prior = (
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.0, 3.0),
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0),
        R₀ = amplitude_prior
    )
    fixed = Base.structdiff(FIDUCIALS, NamedTuple{(:zpeak, :R₀)}(FIDUCIALS))
    observed = forward_model(
        model, polarization_power, SAMPLES, FIDUCIALS).spectral_density

    # Scale the noise to the signal. An arbitrary PSD here puts ρ at 1e21, where the
    # conditional posterior is a delta function on any grid and the identity below is
    # testing floating-point noise rather than the marginalization. σ = 5% of the signal
    # puts ρ in the tens, which is also the regime the marginalization exists for.
    σ_target = 0.05 * maximum(observed)
    eff_psd = fill(
        σ_target * sqrt(2 * year_to_second(observation_time) *
             frequency_bin_width(frequencies)),
        length(frequencies))

    general = astrosgwb_importance_turing_model(
        model, polarization_power, SAMPLES, bns_hyperprior(full_prior), observed, frequencies, eff_psd,
        observation_time, AnalyticInclination()) | fixed
    # The marginalized model's prior omits `R₀` entirely -- it is pinned inside the model,
    # not conditioned at the call site, which is the one place the two models' plumbing
    # genuinely differs.
    marginalized = astrosgwb_amplitude_marginalized_turing_model(
        model, polarization_power, SAMPLES,
        bns_hyperprior_amplitude_marginalized(
            Base.structdiff(full_prior, (; R₀ = nothing)), Val(:R₀)),
        observed, frequencies, eff_psd, observation_time, AnalyticInclination(),
        (; R₀ = FIDUCIALS.R₀), bns_amplitude_scalings(:R₀).amplitude_fn,
        amplitude_prior, grid) | fixed

    θ = (; zpeak = TARGET.zpeak)
    log_integrand = [Turing.logjoint(general, merge(θ, (; R₀ = φ))) for φ in grid]
    m = maximum(log_integrand)
    y = exp.(log_integrand .- m)
    reference = m + log(sum(0.5 .* (y[1:(end - 1)] .+ y[2:end]) .* diff(collect(grid))))

    @test Turing.logjoint(marginalized, θ) ≈ reference rtol = 1.0e-10
end
