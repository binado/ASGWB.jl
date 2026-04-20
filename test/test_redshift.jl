using HDF5
using Test

@testset "redshift parity" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    h5open(fixture_path, "r") do file
        group = file["redshift_case"]
        theta = HyperParameters(;
            H0=Float64(read(group["theta/H0"])),
            Omega_m=Float64(read(group["theta/Omega_m"])),
            gamma=Float64(read(group["theta/gamma"])),
            kappa=Float64(read(group["theta/kappa"])),
            z_peak=Float64(read(group["theta/z_peak"])),
        )
        spec = RedshiftPriorSpec(
            parse_redshift_prior_family(String(read(group["spec/family"]))),
            Float64(read(group["spec/z_min"])),
            Float64(read(group["spec/z_max"])),
            Int(read(group["spec/num_interp"])),
            nothing,
        )
        sample_z = vec(Float64.(read(group["sample_z"])))
        expected_log_prob = vec(Float64.(read(group["log_prob"])))
        expected_integral = Float64(read(group["redshift_integral"]))

        bundle = build_redshift_grid_bundle(theta, spec)

        # Fixture expected values were computed against the Python trapezoid-based bundle
        # norm; Julia now uses composite Simpson, so the tolerances reflect the trapezoid
        # vs Simpson discretization gap rather than numerical precision.
        @test log_prob_from_bundle.(sample_z, Ref(bundle)) ≈ expected_log_prob rtol = 5e-3
        @test bundle.norm ≈ expected_integral rtol = 5e-3
    end
end
