using Test
using Distributions
using CBCDistributions

@testset "PopulationModel interface — TestPop" begin
    pop = TestPop()
    @test hyperparameters(pop) == (:α, :β)

    hp = population_hyperprior(pop)
    @test hp isa Distributions.ProductNamedTupleDistribution
    @test keys(hp.dists) == (:α, :β)

    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.5, β = 1.5)
    sep = single_event_prior(pop, cosmo, Λ)
    @test sep isa Distributions.ProductNamedTupleDistribution
    @test :x in keys(sep.dists)
    @test :y in keys(sep.dists)
end

@testset "full_hyperparameters and merge_hyperpriors" begin
    pop = TestPop()
    @test full_hyperparameters(ModifiedPropagation{LambdaCDM}, pop) ==
          (:H0, :Ωm, :Ξ₀, :Ξₙ, :α, :β)

    hp = merge_hyperpriors(
        cosmology_hyperprior(ModifiedPropagation{LambdaCDM}),
        population_hyperprior(pop)
    )
    @test hp isa Distributions.ProductNamedTupleDistribution
    @test keys(hp.dists) == (:H0, :Ωm, :Ξ₀, :Ξₙ, :α, :β)

    @test full_hyperparameters(LambdaCDM, pop) == (:H0, :Ωm, :α, :β)
end

@testset "cosmology_hyperprior for cosmology types" begin
    hp_lcdm = cosmology_hyperprior(LambdaCDM)
    @test keys(hp_lcdm.dists) == (:H0, :Ωm)

    hp_w0 = cosmology_hyperprior(W0CDM)
    @test keys(hp_w0.dists) == (:H0, :Ωm, :w0)

    hp_cpl = cosmology_hyperprior(W0WaCDM)
    @test keys(hp_cpl.dists) == (:H0, :Ωm, :w0, :wa)

    hp_mod = cosmology_hyperprior(ModifiedPropagation{LambdaCDM})
    @test keys(hp_mod.dists) == (:H0, :Ωm, :Ξ₀, :Ξₙ)

    hp_mod_w0 = cosmology_hyperprior(ModifiedPropagation{W0CDM})
    @test keys(hp_mod_w0.dists) == (:H0, :Ωm, :w0, :Ξ₀, :Ξₙ)
end

@testset "canonical_hyperparameters" begin
    order = (:H0, :Ωm, :α, :β)
    Λ_unordered = (α = 0.5f0, H0 = 67.0, β = 1.5, Ωm = 0.315)

    Λc = canonical_hyperparameters(order, Λ_unordered)
    @test keys(Λc) == order
    @test Λc.H0 isa Float64
    @test Λc.α isa Float64

    Λc_raw = canonical_hyperparameters(order, Λ_unordered; eltype = nothing)
    @test Λc_raw.H0 === 67.0
    @test Λc_raw.α === 0.5f0

    @test_throws ArgumentError canonical_hyperparameters(order, (H0 = 1.0,))
    @test_throws ArgumentError canonical_hyperparameters(order, (;
        (k => 1.0 for k in order)..., extra = 0.0))
end

@testset "validate_hyperparameters" begin
    order = (:H0, :Ωm)
    ok = (H0 = 67.0, Ωm = 0.315)
    @test validate_hyperparameters(order, ok) === nothing

    @test_throws ArgumentError validate_hyperparameters(order, (Ωm = 0.315, H0 = 67.0))
    @test_throws ArgumentError validate_hyperparameters(order, (H0 = 67.0,))
end

@testset "batched_logpdf on ProductNamedTupleDistribution" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    prior = single_event_prior(pop, cosmo, Λ)
    samples = (x = [0.1, 0.5], y = [1.1, 1.7])
    lp = batched_logpdf(prior, samples)
    @test length(lp) == 2
    @test all(isfinite, lp)

    lp_ref = [logpdf(prior, (; x = samples.x[i], y = samples.y[i])) for i in 1:2]
    @test lp ≈ lp_ref
end

@testset "component_logpdfs matches batched_logpdf component sums" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    prior = single_event_prior(pop, cosmo, Λ)
    samples = (x = [0.1, 0.5, 0.7], y = [1.1, 1.7, 0.2])

    lps = component_logpdfs(prior, samples)
    @test keys(lps) == keys(prior.dists)
    @test lps.x .+ lps.y ≈ batched_logpdf(prior, samples)
end

@testset "component_logpdfs rejects mismatched sample field lengths" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    prior = single_event_prior(pop, cosmo, Λ)
    samples = (x = [0.1, 0.5], y = [1.1])

    @test_throws ArgumentError component_logpdfs(prior, samples)
end

@testset "logprobdiff egal skip rejects out-of-support proposal logpdfs" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    samples = (x = [0.1, 1.5], y = [1.1, 1.7])  # x[2] outside Uniform(0, α)

    proposal = single_event_prior(pop, cosmo, Λ)
    proposal_logprob = component_logpdfs(proposal, samples)
    @test !all(isfinite, proposal_logprob.x)

    prior_same = single_event_prior(pop, cosmo, Λ)
    diff = logprobdiff(pop, prior_same, proposal, proposal_logprob, samples)
    ref = batched_logpdf(prior_same, samples) .- batched_logpdf(proposal, samples)
    @test diff[1] ≈ ref[1]
    @test isnan(diff[2]) && isnan(ref[2])
end

@testset "logprobdiff default: egal skip and two-sided reference" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ_fid = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.6, β = 1.4)
    samples = (x = [0.1, 0.5], y = [1.1, 0.3])

    proposal = single_event_prior(pop, cosmo, Λ_fid)
    proposal_logprob = component_logpdfs(proposal, samples)

    # Same hyperparameters: every component is egal to the proposal's, so the diff is
    # exactly zero (skipped, never evaluated).
    prior_same = single_event_prior(pop, cosmo, Λ_fid)
    @test prior_same.dists.x === proposal.dists.x
    diff_same = logprobdiff(pop, prior_same, proposal, proposal_logprob, samples)
    @test diff_same == zeros(length(samples.x))

    # Different hyperparameters: nothing is egal, so the diff matches the full
    # two-sided batched_logpdf reference.
    prior = single_event_prior(pop, cosmo, Λ)
    diff = logprobdiff(pop, prior, proposal, proposal_logprob, samples)
    @test diff ≈ batched_logpdf(prior, samples) .- batched_logpdf(proposal, samples)

    # Convenience wrapper (proposal logpdfs computed on the fly) agrees.
    @test logprobdiff(pop, prior, proposal, samples) ≈ diff
end

@testset "logprobdiff per-component override via Val{key}" begin
    pop = SkipYPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ_fid = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.6, β = 1.4)
    samples = (x = [0.1, 0.5], y = [1.1, 0.3])

    proposal = single_event_prior(pop, cosmo, Λ_fid)
    proposal_logprob = component_logpdfs(proposal, samples)
    prior = single_event_prior(pop, cosmo, Λ)

    # Only the :x component contributes; :y is skipped by the override even though the
    # target and proposal distributions differ.
    diff = logprobdiff(pop, prior, proposal, proposal_logprob, samples)
    x_only = component_logpdfs(prior, samples).x .- proposal_logprob.x
    @test diff ≈ x_only
end
