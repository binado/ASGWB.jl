module BackgroundCosmology

using Trapezoid: trapz, cumtrapz

export AbstractCosmology, LambdaCDM, W0CDM, W0WaCDM,
       AbstractPropagation, GR, ModifiedPropagation,
       E, dark_energy_eos, de_density_ratio,
       hubble_constant_si, H0, Ωm,
       cosmology,
       cosmology_type, SUPPORTED_COSMOLOGIES,
       propagation,
       propagation_type, propagation_config_name, SUPPORTED_PROPAGATIONS,
       comoving_distance, luminosity_distance, differential_comoving_volume,
       distance_and_volume_grid, trapz, cumtrapz,
       gw_em_distance_ratio,
       apply_gw_distance_correction, apply_gw_distance_correction!

include("conversion.jl")
include("model.jl")
include("distance.jl")
include("modified_propagation.jl")

end # module
