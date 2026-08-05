using Test
using AstroSGWB
using AstroSGWBImportanceModels
using AstroSGWBInference
using Cosmology
using Distributions: Uniform, product_distribution
using ForwardDiff
using Turing

const FIDUCIALS = (
    H0 = 67.0,
    Ωm = 0.315,
    Ξ₀ = 1.0,
    Ξₙ = 0.0,
    γ = 2.7,
    κ = 3.0,
    zpeak = 2.5
)

const TARGET = (
    H0 = 70.0,
    Ωm = 0.3,
    Ξ₀ = 1.1,
    Ξₙ = 0.2,
    γ = 2.9,
    κ = 3.1,
    zpeak = 2.2
)

const SAMPLES = (
    redshift = [0.1, 0.2],
    luminosity_distance = [430.0, 880.0]
)

function prepared(samples = SAMPLES; C = LambdaCDM, P = ModifiedPropagation,
        fiducials = FIDUCIALS)
    return prepare_bns_madau_dickinson_model(
        samples,
        fiducials,
        C,
        P;
        local_merger_rate = 161.0,
        observation_time = 1.0
    )
end

@testset "BNS Madau–Dickinson hyperparameters" begin
    @test bns_madau_dickinson_hyperparameters(LambdaCDM, GR) ==
          (:H0, :Ωm, :γ, :κ, :zpeak)
    @test bns_madau_dickinson_hyperparameters(LambdaCDM, ModifiedPropagation) ==
          (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    @test bns_madau_dickinson_hyperparameters(W0CDM, GR) ==
          (:H0, :Ωm, :w0, :γ, :κ, :zpeak)
    @test bns_madau_dickinson_hyperparameters(W0CDM, ModifiedPropagation) ==
          (:H0, :Ωm, :w0, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    @test AstroSGWBInference.hyperparameters(prepared()) ==
          bns_madau_dickinson_hyperparameters(LambdaCDM, ModifiedPropagation)
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
    @test model.local_merger_rate === 161.0
    @test model.observation_time === 1.0
    @test length(model.z_grid) == length(DEFAULT_Z_GRID)
    @test length(model.proposal_log_pdf) == length(SAMPLES.redshift)
    @test all(isfinite, model.proposal_log_pdf)
    @test model.proposal_log_pdf ≈ [-6.274110509399128, -4.919648956007439]

    rate, log_weights = merger_rate_and_log_weights(model, TARGET, SAMPLES)
    @test rate ≈ 0.031115713391297647 rtol = 1.0e-13
    # Refrozen when S5/S6 moved the hot path onto `distance_and_volume_grid` +
    # `GridInterpolator`. `proposal_log_pdf` and `rate` above are bit-unchanged (they
    # depend only on the density and its trapezoid normalizer); these shifted because
    # `d_L` is now plain-linearly interpolated off the grid rather than reconstructed from
    # the exact within-cell antiderivative of 1/E — a deliberate change, matching the
    # Python stack. Linear-interpolation error is O(Δz²·f″) absolute, but d_L → 0 as
    # z → 0, so the *relative* error carries a 1/z factor: measured Δ = -1.524e-2 at
    # z = 0.1 and -8.29e-3 at z = 0.2, falling to ~-3.7e-4 at z = 1 and ~-8e-6 at z = 5.
    @test log_weights ≈ [-0.08905516675812283, -0.15599248526634377] rtol = 1.0e-12
    @test size(log_weights) == size(SAMPLES.redshift)
    @test all(isfinite, log_weights)

    @test_throws DimensionMismatch merger_rate_and_log_weights(
        model, TARGET, (redshift = [0.1], luminosity_distance = [430.0]))
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
    _, w = merger_rate_and_log_weights(model, FIDUCIALS, grid_samples)
    @test maximum(abs, w) < 1e-14

    # With the production sample adapter the residual is NOT zero: it synthesizes `d_L`
    # with `quadgk` while the hot path reads a 256-point grid. Pre-existing systematic
    # (~2e-2 in log-weight at z = 0.1), unchanged by S5/S6 and tracked separately.
    _,
    w_quadgk = merger_rate_and_log_weights(
        model, FIDUCIALS,
        bns_samples_from_catalog((redshift = z,), LambdaCDM, FIDUCIALS))
    @test 1e-3 < maximum(abs, w_quadgk) < 1e-1
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
    _, w_gr = merger_rate_and_log_weights(m_gr, TARGET, SAMPLES)
    _, w_mod = merger_rate_and_log_weights(m_mod, TARGET, SAMPLES)
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
    empty_weights = merger_rate_and_log_weights(
        empty_model, Λ_dual, empty_samples)
    one_rate, one_weights = merger_rate_and_log_weights(one_model, Λ_dual, one_sample)

    @test isfinite(empty_rate)
    @test isfinite(one_rate)
    @test isempty(empty_weights)
    @test eltype(empty_weights) <: ForwardDiff.Dual
    @test length(one_weights) == 1
    @test all(isfinite, one_weights)
    @test eltype(one_weights) <: ForwardDiff.Dual
end

@testset "concrete adapter integrates with Turing" begin
    model = prepared()
    fluxes = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    observation = ObservationContext(
        [0.0, 20.0, 40.0],
        [Inf, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        BitVector([false, true, true]),
        1.0
    )
    prior = product_distribution((
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.0, 3.0),
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    ))
    turing_model = build_turing_model(
        model, fluxes, SAMPLES, FIDUCIALS, observation, prior)

    @test turing_model !== nothing
    @test isfinite(Turing.logjoint(turing_model, FIDUCIALS))
end
