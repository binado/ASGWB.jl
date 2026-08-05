module CBCDistributions

import Cosmology
import Cosmology: AbstractCosmology, AbstractPropagation, CosmologyCache, E,
                  CumulativeIntegral1D, cdf, interpolate, normalizer,
                  hyperparameters, propagation_hyperparameters

export PopulationModel, hyperparameters, single_event_prior,
       full_hyperparameters,
       canonical_hyperparameters, validate_hyperparameters
export MadauDickinsonSourceFrame, source_frame_distribution, redshift_prior, DEFAULT_Z_GRID
export DefaultBBHPrimaryMass, DefaultBBHMassPair, planck_taper
export JULIAN_YEAR_SEC, year_to_second, second_to_year

include("utils.jl")
include("mass/uniform.jl")
include("mass/broken_power_law_plus_two_peaks.jl")
include("spins/aligned.jl")
include("redshift.jl")
include("population_model.jl")

end # module
