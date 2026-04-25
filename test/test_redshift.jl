using HDF5
using Test

function _reference_redshift_bundle(
        h,
        spec::RedshiftPriorSpec,
        z_grid::AbstractVector{<:Real}
)
    sfn = z -> madau_dickinson_source_frame_distribution(
        z;
        γ = h.γ,
        κ = h.κ,
        zpeak = h.zpeak
    )
    inv_E = w -> inv(E(w, h.Ωm))
    distance = CumulativeIntegral1D(z_grid, inv_E)
    d_h = ASGWB.SPEED_OF_LIGHT_KM_S / h.H0
    pdf_integrand = let dist = distance, sf = sfn, Ωm′ = h.Ωm, dh = d_h
        function (w)
            d_c = dh * ASGWB.cdf(dist, w)
            dvc_dz = dh * d_c^2 / E(w, Ωm′)
            return detector_frame_merger_rate_density(w, dvc_dz, sf(w))
        end
    end
    pdf = CumulativeIntegral1D(z_grid, pdf_integrand)
    return RedshiftBundle(distance, pdf)
end

@testset "redshift parity" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    h5open(fixture_path, "r") do file
        group = file["redshift_case"]
        theta = HyperParameters(;
            H0 = Float64(read(group["theta/H0"])),
            Ωm = Float64(read(group["theta/Omega_m"])),
            γ = Float64(read(group["theta/gamma"])),
            κ = Float64(read(group["theta/kappa"])),
            zpeak = Float64(read(group["theta/z_peak"]))
        )
        spec = RedshiftPriorSpec(
            parse_redshift_prior_family(String(read(group["spec/family"]))),
            Float64(read(group["spec/z_min"])),
            Float64(read(group["spec/z_max"])),
            Int(read(group["spec/num_interp"])),
            nothing
        )
        sample_z = vec(Float64.(read(group["sample_z"])))
        expected_log_prob = vec(Float64.(read(group["log_prob"])))
        expected_integral = Float64(read(group["redshift_integral"]))

        bundle = build_redshift_grid_bundle(theta, spec)
        z_grid = ASGWB.redshift_grid(spec)
        reference = _reference_redshift_bundle(theta, spec, z_grid)

        @test bundle.distance.y ≈ reference.distance.y rtol = 1e-12
        @test bundle.distance.cumulative ≈ reference.distance.cumulative rtol = 1e-12
        @test bundle.pdf.y ≈ reference.pdf.y rtol = 1e-12
        @test bundle.pdf.cumulative ≈ reference.pdf.cumulative rtol = 1e-12
        @test log_prob_from_bundle.(sample_z, Ref(bundle)) ≈
              log_prob_from_bundle.(sample_z, Ref(reference)) rtol = 1e-12
        @test ASGWB.redshift_integral(bundle) ≈ ASGWB.redshift_integral(reference) rtol = 1e-12

        # Fixture expected values were generated from the Python implementation; keep a
        # modest tolerance here since this test is cross-language parity rather than an
        # exact same-code-path regression.
        @test log_prob_from_bundle.(sample_z, Ref(bundle)) ≈ expected_log_prob rtol = 5e-3
        @test ASGWB.redshift_integral(bundle) ≈ expected_integral rtol = 5e-3
    end
end

@testset "sample interpolant parity" begin
    theta = HyperParameters(;
        H0 = 67.0,
        Ωm = 0.315,
        γ = 2.7,
        κ = 3.0,
        zpeak = 2.5
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    z_grid = ASGWB.redshift_grid(spec)
    bundle = build_redshift_grid_bundle(theta, spec, z_grid)
    samples = [spec.z_min, 0.37, 1.2, spec.z_max]
    interp = ASGWB.SampleInterpolant(samples, z_grid)

    @test interp.bin_idx[1] == 1
    @test interp.t[1] == 0.0
    @test interp.bin_idx[end] == length(z_grid) - 1
    @test interp.t[end] == 1.0

    @test ASGWB.interpolate_at_samples(bundle.pdf.y, interp, z_grid) ≈
          interpolate.(Ref(bundle.pdf), samples) rtol = 1e-12
    @test ASGWB.cdf_at_samples(bundle.distance.cumulative, bundle.distance.y, interp, z_grid) ≈
          cdf.(Ref(bundle.distance), samples) rtol = 1e-12
    @test log_prob_from_bundle(samples, bundle, interp) ≈
          log_prob_from_bundle.(samples, Ref(bundle)) rtol = 1e-12

    @test_throws ArgumentError ASGWB.SampleInterpolant([spec.z_min - 1e-6], z_grid)
    @test_throws ArgumentError ASGWB.SampleInterpolant([spec.z_max + 1e-6], z_grid)
end
