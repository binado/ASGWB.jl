using HDF5

const JULIA_IMPORTANCE_CACHE_FORMAT_NAME = "asgwb.julia.importance_cache"
const JULIA_IMPORTANCE_CACHE_FORMAT_VERSION = 1

_as_string(value::AbstractString) = String(value)
_as_string(value::AbstractVector{UInt8}) = String(copy(value))
_as_string(value) = string(value)

function _read_string_vector(dataset)::Vector{String}
    values = read(dataset)
    return [_as_string(value) for value in values]
end

function _read_float_vector(dataset, name::AbstractString)::Vector{Float64}
    values = vec(Float64.(read(dataset)))
    isempty(values) && throw(ArgumentError("$(name) must not be empty"))
    return values
end

function _read_bool_vector(dataset)::BitVector
    return BitVector(vec(Bool.(read(dataset))))
end

function _read_float_matrix(dataset, name::AbstractString)::Matrix{Float64}
    values = permutedims(Array{Float64}(read(dataset)))
    ndims(values) == 2 || throw(ArgumentError("$(name) must be a 2D dataset"))
    return values
end

function _read_float_scalar_dataset(group::HDF5.Group, key::AbstractString)::Float64
    return Float64(read(group[key]))
end

function _require_child(group, name::AbstractString)
    haskey(group, name) || throw(ArgumentError("missing required HDF5 entry: $(name)"))
    return group[name]
end

function _read_attr(attrs, name::AbstractString)
    haskey(attrs, name) || throw(ArgumentError("missing required HDF5 attribute: $(name)"))
    return read(attrs[name])
end

function _read_optional_string(group, key::AbstractString)::Union{String,Nothing}
    haskey(group, key) || return nothing
    raw = read(group[key])
    text = _as_string(raw)
    return isempty(text) ? nothing : text
end

function _read_redshift_prior_spec(group)::RedshiftPriorSpec
    spec_group = _require_child(group, "redshift_prior_spec")
    family_str = _as_string(read(_require_child(spec_group, "family")))
    return RedshiftPriorSpec(
        parse_redshift_prior_family(family_str),
        Float64(read(_require_child(spec_group, "z_min"))),
        Float64(read(_require_child(spec_group, "z_max"))),
        Int(read(_require_child(spec_group, "num_interp"))),
        _read_optional_string(spec_group, "time_delay_model"),
    )
end

const _CACHE_HYPERPARAMETER_KEYS = ("H0", "Omega_m", "chi0", "chin")

function _read_hyperparameters(group::HDF5.Group)::HyperParameters
    for k in _CACHE_HYPERPARAMETER_KEYS
        haskey(group, k) || throw(ArgumentError("missing hyperparameter $(k)"))
    end
    for k in keys(group)
        kn = String(k)
        kn in _CACHE_HYPERPARAMETER_KEYS ||
            throw(ArgumentError("unknown hyperparameter $(kn)"))
    end
    return HyperParameters(;
        H0=_read_float_scalar_dataset(group, "H0"),
        Omega_m=_read_float_scalar_dataset(group, "Omega_m"),
        chi0=_read_float_scalar_dataset(group, "chi0"),
        chin=_read_float_scalar_dataset(group, "chin"),
    )
end

"""
    load_cache(path::AbstractString) -> ImportanceSamplingProblem

Read a Julia-native HDF5 importance cache written with format
`asgwb.julia.importance_cache` and return an [`ImportanceSamplingProblem`](@ref).

This is a convenience wrapper around disk I/O; equivalent in-memory problems can
be built with [`importance_sampling_problem`](@ref).
"""
function load_cache(path::AbstractString)::ImportanceSamplingProblem
    return h5open(path, "r") do file
        attrs = attributes(file)
        format_name = _as_string(_read_attr(attrs, "format_name"))
        format_name == JULIA_IMPORTANCE_CACHE_FORMAT_NAME || throw(
            ArgumentError(
                "unsupported cache format $(format_name), expected $(JULIA_IMPORTANCE_CACHE_FORMAT_NAME)",
            ),
        )

        format_version = Int(_read_attr(attrs, "format_version"))
        format_version == JULIA_IMPORTANCE_CACHE_FORMAT_VERSION || throw(
            ArgumentError(
                "unsupported cache format version $(format_version), expected $(JULIA_IMPORTANCE_CACHE_FORMAT_VERSION)",
            ),
        )

        intrinsic_site_order = _read_string_vector(
            _require_child(file, "intrinsic_site_order"),
        )
        proposal_log_prob = _read_float_vector(
            _require_child(file, "proposal_log_prob"),
            "proposal_log_prob",
        )
        proposal_intrinsic_vector = _read_float_matrix(
            _require_child(file, "proposal_intrinsic_vector"),
            "proposal_intrinsic_vector",
        )
        cached_flux_over_dgw2 = _read_float_matrix(
            _require_child(file, "cached_flux_over_dgw2"),
            "cached_flux_over_dgw2",
        )
        dgw_fid_sq = _read_float_vector(_require_child(file, "dgw_fid_sq"), "dgw_fid_sq")
        frequencies = _read_float_vector(
            _require_child(file, "frequencies"),
            "frequencies",
        )
        covariance = _read_float_vector(_require_child(file, "covariance"), "covariance")
        sgwb_scale = _read_float_vector(_require_child(file, "sgwb_scale"), "sgwb_scale")
        in_band_mask = _read_bool_vector(_require_child(file, "in_band_mask"))
        fiducial_spectral_density = _read_float_vector(
            _require_child(file, "fiducial_spectral_density"),
            "fiducial_spectral_density",
        )

        proposal_samples = Dict{String,Vector{Float64}}()
        proposal_samples_group = _require_child(file, "proposal_samples")
        for key in intrinsic_site_order
            proposal_samples[key] = _read_float_vector(
                _require_child(proposal_samples_group, key),
                "proposal_samples/$(key)",
            )
        end

        hyperparameters = _read_hyperparameters(_require_child(file, "hyperparameters"))

        redshift_prior_spec = _read_redshift_prior_spec(file)

        n_samples = length(proposal_log_prob)
        length(intrinsic_site_order) == size(proposal_intrinsic_vector, 2) || throw(
            ArgumentError(
                "proposal_intrinsic_vector column count must match intrinsic_site_order length",
            ),
        )
        size(proposal_intrinsic_vector, 1) == n_samples || throw(
            ArgumentError(
                "proposal_intrinsic_vector row count must match proposal_log_prob length",
            ),
        )
        size(cached_flux_over_dgw2, 1) == n_samples || throw(
            ArgumentError(
                "cached_flux_over_dgw2 row count must match proposal_log_prob length",
            ),
        )
        length(dgw_fid_sq) == n_samples || throw(
            ArgumentError("dgw_fid_sq length must match proposal_log_prob length"),
        )
        for key in intrinsic_site_order
            length(proposal_samples[key]) == n_samples || throw(
                ArgumentError(
                    "proposal_samples/$(key) length must match proposal_log_prob length",
                ),
            )
        end

        n_frequencies = length(frequencies)
        size(cached_flux_over_dgw2, 2) == n_frequencies || throw(
            ArgumentError(
                "cached_flux_over_dgw2 column count must match frequencies length",
            ),
        )
        length(covariance) == n_frequencies || throw(
            ArgumentError("covariance length must match frequencies length"),
        )
        length(sgwb_scale) == n_frequencies || throw(
            ArgumentError("sgwb_scale length must match frequencies length"),
        )
        length(in_band_mask) == n_frequencies || throw(
            ArgumentError("in_band_mask length must match frequencies length"),
        )
        length(fiducial_spectral_density) == n_frequencies || throw(
            ArgumentError("fiducial_spectral_density length must match frequencies length"),
        )
        haskey(proposal_samples, "redshift") || throw(
            ArgumentError("proposal_samples must include a redshift entry"),
        )

        proposal = ProposalData(
            intrinsic_site_order,
            proposal_samples,
            proposal_log_prob,
            proposal_intrinsic_vector,
            cached_flux_over_dgw2,
            dgw_fid_sq,
        )

        observation = ObservationConfig(
            frequencies,
            covariance,
            sgwb_scale,
            in_band_mask,
            fiducial_spectral_density,
            Float64(_read_attr(attrs, "observation_time_sec")),
            Float64(_read_attr(attrs, "observation_time_yr")),
        )

        return importance_sampling_problem(
            proposal,
            observation,
            redshift_prior_spec,
            Float64(_read_attr(attrs, "local_merger_rate")),
            Float64(_read_attr(attrs, "redshift_integral_fiducial")),
            hyperparameters,
        )
    end
end
