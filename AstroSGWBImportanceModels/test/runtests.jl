using Test
using AstroSGWB
using AstroSGWBImportanceModels
using AstroSGWBInference
using Cosmology
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
        fiducials = FIDUCIALS)
    return prepare_bns_madau_dickinson_model(samples, fiducials, C, P)
end

@testset "the prepared model is the contract callable" begin
    # S1: the whole model contract is `weights_fn(Λ, samples) -> (rate, log_weights)`.
    # No abstract supertype, no generic function to add methods to -- so this package
    # imports nothing from `AstroSGWBInference` and a plain closure would serve equally.
    model = prepared()
    @test !isempty(methods(model))
    rate, log_weights = model(TARGET, SAMPLES)
    @test rate isa Real
    @test length(log_weights) == length(SAMPLES.redshift)
end

@testset "catalog sample adaptation" begin
    stored = (
        redshift = [0.1, 0.2],
        luminosity_distance = [12.0, 34.0],
        unused = [1, 2]
    )
    adapted = bns_samples_from_catalog(stored, LambdaCDM, FIDUCIALS)
    @test adapted == (redshift = [0.1, 0.2], luminosity_distance = [12.0, 34.0])
    @test adapted.redshift !== stored.redshift
    @test adapted.luminosity_distance !== stored.luminosity_distance

    without_distance = (redshift = [0.1, 0.2], unused = [1, 2])
    synthesized = bns_samples_from_catalog(without_distance, LambdaCDM, FIDUCIALS)
    expected = luminosity_distance.(
        without_distance.redshift, Ref(cosmology(LambdaCDM, FIDUCIALS)))
    @test synthesized.redshift == without_distance.redshift
    @test synthesized.luminosity_distance ≈ expected
    @test all(isfinite, synthesized.luminosity_distance)
    @test all(>(0), synthesized.luminosity_distance)
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
end

@testset "prepare and hot path share one kernel" begin
    model = prepared()
    # `==`, not `≈`: `_bns_grid_terms` is literally the function `prepare` called, so at
    # Λ == fiducials the target density is bit-identical to the cached proposal density.
    # If these ever diverge, every posterior silently acquires a per-sample offset.
    @test AstroSGWBImportanceModels._bns_grid_terms(
        LambdaCDM, FIDUCIALS, model.z_grid, model.interp).log_p ==
          model.proposal_log_pdf

    # The full weight expression vanishes when `d_L_fid` is built through the *same* grid
    # path the hot path uses, so nothing is left but exact cancellation.
    z = SAMPLES.redshift
    d_l_grid = model.interp(
        distance_and_volume_grid(cosmology(LambdaCDM, FIDUCIALS),
        model.z_grid).luminosity_distance)
    grid_samples = (redshift = z, luminosity_distance = d_l_grid)
    _, w = model(FIDUCIALS, grid_samples)
    @test maximum(abs, w) < 1e-14

    # With the production sample adapter the residual is NOT zero: it synthesizes `d_L`
    # with `quadgk` while the hot path reads a 256-point grid. Pre-existing systematic
    # (~4.6e-2 in log-weight at z = 0.1), tracked separately. astrogwb has the same
    # residual for the same reason — its catalog `luminosity_distance` column also does
    # not come from the 256-point grid the weights are evaluated on.
    _,
    w_quadgk = model(FIDUCIALS,
        bns_samples_from_catalog((redshift = z,), LambdaCDM, FIDUCIALS))
    @test 1e-3 < maximum(abs, w_quadgk) < 1e-1
end

@testset "DEFAULT_Z_GRID starts at zero" begin
    # Not cosmetic, and not merely a Julia-internal choice. Comoving distance is
    # accumulated by trapezoidal integration along the grid assuming `d_c(grid[1]) = 0`,
    # so a non-zero lower bound silently omits `∫₀^{z_min} dz/E` from *every* distance.
    # The former `1e-3` bound cost ≈ 4.5 Mpc: −1.0% in d_L at z = 0.1, −20% at z = 0.005.
    # `astrogwb.cosmology.distance_and_volume_grid` documents the same requirement, so
    # this is also what keeps the two repos' weights comparable.
    @test first(DEFAULT_Z_GRID) == 0.0
    @test last(DEFAULT_Z_GRID) == 20.0
    @test length(DEFAULT_Z_GRID) == 256

    # The consequence, stated directly: with a 1e-3 floor the interpolated d_L at low z
    # is short by ~1%, which is 2e-2 in log-weight.
    cosmo = cosmology(LambdaCDM, FIDUCIALS)
    truncated = collect(LinRange(1e-3, 20.0, 256))
    zs = [0.1]
    d_zero = GridInterpolator(zs, DEFAULT_Z_GRID)(
        distance_and_volume_grid(cosmo, DEFAULT_Z_GRID).luminosity_distance)
    d_trunc = GridInterpolator(zs, truncated)(
        distance_and_volume_grid(cosmo, truncated).luminosity_distance)
    @test only(d_trunc) < only(d_zero)
    @test (only(d_zero) - only(d_trunc)) / only(d_zero) ≈ 0.0104 atol = 5e-4
end

@testset "S11 fiducial GW-distance reference" begin
    fid_gr = merge(FIDUCIALS, (Ξ₀ = 1.0, Ξₙ = 0.0))
    fid_mod = merge(FIDUCIALS, (Ξ₀ = 1.4, Ξₙ = 0.7))
    prop_fid = propagation(ModifiedPropagation, fid_mod)
    z = SAMPLES.redshift
    fluxes = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]

    m_gr, m_mod = prepared(; fiducials = fid_gr), prepared(; fiducials = fid_mod)
    @test m_gr.log_Ξ_fid == zeros(length(z))
    # Isolates the change to log_Ξ_fid: the proposal density is propagation-independent.
    @test m_mod.proposal_log_pdf == m_gr.proposal_log_pdf
    _, w_gr = m_gr(TARGET, SAMPLES)
    _, w_mod = m_mod(TARGET, SAMPLES)
    @test w_mod ≈ w_gr .+ 2 .* log.(gw_em_distance_ratio.(z, Ref(prop_fid)))

    # The load-bearing invariant, stated on the physical contraction:
    # corrected fluxes + Ξ_fid weights == uncorrected fluxes + no-Ξ_fid weights.
    corrected = apply_gw_distance_correction(fluxes, z, prop_fid)
    @test corrected ≉ fluxes                                   # anti-vacuity guard
    @test corrected * exp.(w_mod) ≈ fluxes * exp.(w_gr) rtol = 1e-13

    @test apply_gw_distance_correction!(fluxes, z, GR()) === fluxes
    @test apply_gw_distance_correction(fluxes, z,
        propagation(ModifiedPropagation, fid_gr)) ≈ fluxes
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
    # exact shape `merge(constants, Λ_sampled)` produces on every gradient evaluation.
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
    fluxes = Float64[1.0 1.5; 2.0 2.5]
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
        zpeak = Uniform(0.05, 10.0)
    )
    # `R₀` is fixed via `constants` rather than sampled -- the production default. The
    # prior declares the sampled names; `constants` supplies the rest of `Λ`.
    turing_model = build_turing_model(
        model, fluxes, SAMPLES, FIDUCIALS, frequencies, eff_psd, observation_time, prior;
        constants = (; R₀ = FIDUCIALS.R₀))

    @test turing_model !== nothing
    @test isfinite(Turing.logjoint(turing_model, FIDUCIALS))

    # And the opt-in: adding `R₀` to the prior makes it a sampled variable, with no
    # change anywhere else.
    sampling_R₀ = build_turing_model(
        model, fluxes, SAMPLES, FIDUCIALS, frequencies, eff_psd, observation_time,
        merge(prior, (; R₀ = Uniform(10.0, 1000.0))))
    @test isfinite(Turing.logjoint(sampling_R₀, FIDUCIALS))
    @test Set(Symbol.(keys(Turing.DynamicPPL.VarInfo(sampling_R₀)))) ==
          Set(keys(FIDUCIALS))
end
