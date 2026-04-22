using HDF5

"""Root HDF5 attribute: shell command used to generate the importance cache."""
const IMPORTANCE_CACHE_COMMAND_ATTR = "command"

"""Root HDF5 attribute: git revision (object id) of the generator codebase."""
const IMPORTANCE_CACHE_GIT_REVISION_ATTR = "git_revision"

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

function _read_in_band_mask(dataset)::BitVector
    raw = read(dataset)
    if raw isa BitVector
        return raw
    elseif raw isa AbstractVector{Bool}
        return BitVector(vec(raw))
    else
        v = vec(raw)
        if eltype(v) <: Integer
            return BitVector(Int.(v) .!= 0)
        end
        try
            return BitVector(Bool.(v))
        catch
            throw(
                ArgumentError(
                "in_band_mask: unsupported HDF5 element type $(eltype(v)); " *
                "expected bool, integer, or HDF5 enum compatible with Bool",
            ),
            )
        end
    end
end

"""
    _read_hdf5_col_sample_matrix(dataset, name, n_samples, n_cols) -> Matrix{Float64}

Return `Matrix{Float64}` of shape `(n_samples, n_cols)` (rows = samples, columns = frequency
bins or intrinsic sites).

**On-disk contract** (HDF5 dataspace / `h5dump` order): extent is always `(n_cols, n_samples)`
(first index runs over frequency or intrinsic parameter, second over proposal sample).

`HDF5.read` may return that layout as `(n_cols, n_samples)` or already transposed to
`(n_samples, n_cols)` depending on the writer and how extents map into Julia column-major
matrices. When it returns `(n_cols, n_samples)`, we apply `permutedims` to obtain sample rows;
when it already returns `(n_samples, n_cols)`, we use it as-is.
"""
function _read_hdf5_col_sample_matrix(
        dataset,
        name::AbstractString,
        n_samples::Int,
        n_cols::Int
)::Matrix{Float64}
    raw = Array{Float64}(read(dataset))
    ndims(raw) == 2 || throw(ArgumentError("$(name) must be a 2D dataset"))
    if size(raw) == (n_cols, n_samples)
        return Matrix(permutedims(raw))
    elseif size(raw) == (n_samples, n_cols)
        return raw
    else
        throw(
            ArgumentError(
            "$(name): HDF5 extent contract is ($n_cols, $n_samples) = (n_columns, n_samples); " *
            "after read expected size ($n_cols, $n_samples) or ($n_samples, $n_cols), got $(size(raw))",
        ),
        )
    end
end

"""
    _read_hdf5_freq_sample_matrix(dataset, name, n_samples, n_freq) -> Matrix{Float64}

Variant of [`_read_hdf5_col_sample_matrix`](@ref) for hot-loop arrays that keep the
column-major-friendly `(n_freq, n_samples)` layout in memory. The on-disk extent
is the same `(n_freq, n_samples)`; when `HDF5.read` returns that shape we use it
as-is, when it returns the transpose we apply `permutedims`.
"""
function _read_hdf5_freq_sample_matrix(
        dataset,
        name::AbstractString,
        n_samples::Int,
        n_freq::Int
)::Matrix{Float64}
    raw = Array{Float64}(read(dataset))
    ndims(raw) == 2 || throw(ArgumentError("$(name) must be a 2D dataset"))
    if size(raw) == (n_freq, n_samples)
        return raw
    elseif size(raw) == (n_samples, n_freq)
        return Matrix(permutedims(raw))
    else
        throw(
            ArgumentError(
            "$(name): HDF5 extent contract is ($n_freq, $n_samples) = (n_columns, n_samples); " *
            "after read expected size ($n_freq, $n_samples) or ($n_samples, $n_freq), got $(size(raw))",
        ),
        )
    end
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

function _read_nonempty_string_attr(attrs, name::AbstractString)::String
    text = strip(_as_string(_read_attr(attrs, name)))
    isempty(text) &&
        throw(ArgumentError("HDF5 attribute $(repr(name)) must be a non-empty string"))
    return text
end

function _read_optional_string(group, key::AbstractString)::Union{String, Nothing}
    haskey(group, key) || return nothing
    raw = read(group[key])
    text = _as_string(raw)
    return isempty(text) ? nothing : text
end

function _validate_proposal_samples_source_type(g::HDF5.Group)
    attrs = attributes(g)
    haskey(attrs, PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR) || throw(
        ArgumentError(
        "missing required HDF5 attribute proposal_samples/$(PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR)",
    ),
    )
    st = strip(_as_string(read(attrs[PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR])))
    st == PROPOSAL_SAMPLES_SOURCE_TYPE_BNS || throw(
        ArgumentError(
        "unsupported proposal_samples/$(PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR)=$(repr(st)); " *
        "only $(repr(PROPOSAL_SAMPLES_SOURCE_TYPE_BNS)) is implemented",
    ),
    )
    return nothing
end

function _read_redshift_prior_spec(group)::RedshiftPriorSpec
    spec_group = _require_child(group, "redshift_prior_spec")
    family_str = _as_string(read(_require_child(spec_group, "family")))
    return RedshiftPriorSpec(
        parse_redshift_prior_family(family_str),
        Float64(read(_require_child(spec_group, "z_min"))),
        Float64(read(_require_child(spec_group, "z_max"))),
        Int(read(_require_child(spec_group, "num_interp"))),
        _read_optional_string(spec_group, "time_delay_model")
    )
end

const _CACHE_FIDUCIAL_KEYS = ("H0", "Omega_m", "chi0", "chin")
const _CACHE_OPTIONAL_POPULATION_KEYS = ("gamma", "kappa", "z_peak", "lamb")

function _read_optional_float_scalar(
        group::HDF5.Group,
        key::AbstractString
)::Union{Nothing, Float64}
    haskey(group, key) || return nothing
    return Float64(read(group[key]))
end

function _merge_population_scalar(
        hyper::HDF5.Group,
        spec::HDF5.Group,
        key::AbstractString
)::Union{Nothing, Float64}
    v_h = _read_optional_float_scalar(hyper, key)
    v_s = _read_optional_float_scalar(spec, key)
    if v_h !== nothing && v_s !== nothing && v_h != v_s
        throw(
            ArgumentError(
            "inconsistent $(key) between hyperparameters ($(v_h)) and " *
            "redshift_prior_spec ($(v_s))",
        ),
        )
    end
    return v_h !== nothing ? v_h : v_s
end

function _read_proposal_fiducial_parameters(
        hyper::HDF5.Group,
        spec::HDF5.Group,
        family::RedshiftPriorFamily
)::ProposalFiducialParameters
    for k in _CACHE_FIDUCIAL_KEYS
        haskey(hyper, k) || throw(ArgumentError("missing hyperparameter $(k)"))
    end
    allowed = Set{String}(
        collect(_CACHE_FIDUCIAL_KEYS) ∪ collect(_CACHE_OPTIONAL_POPULATION_KEYS),
    )
    for k in keys(hyper)
        kn = String(k)
        kn in allowed || throw(ArgumentError("unknown hyperparameter $(kn)"))
    end
    γ = _merge_population_scalar(hyper, spec, "gamma")
    κ = _merge_population_scalar(hyper, spec, "kappa")
    zp = _merge_population_scalar(hyper, spec, "z_peak")
    λ = _merge_population_scalar(hyper, spec, "lamb")
    if family == MadauDickinson
        γ === nothing && throw(
            ArgumentError(
            "Madau–Dickinson cache requires gamma in hyperparameters or redshift_prior_spec",
        ),
        )
        κ === nothing && throw(
            ArgumentError(
            "Madau–Dickinson cache requires kappa in hyperparameters or redshift_prior_spec",
        ),
        )
        zp === nothing && throw(
            ArgumentError(
            "Madau–Dickinson cache requires z_peak in hyperparameters or redshift_prior_spec",
        ),
        )
    else
        λ === nothing && throw(
            ArgumentError(
            "power-law cache requires lamb in hyperparameters or redshift_prior_spec",
        ),
        )
    end
    return ProposalFiducialParameters(;
        H0 = _read_float_scalar_dataset(hyper, "H0"),
        Omega_m = _read_float_scalar_dataset(hyper, "Omega_m"),
        chi0 = _read_float_scalar_dataset(hyper, "chi0"),
        chin = _read_float_scalar_dataset(hyper, "chin"),
        gamma = γ,
        kappa = κ,
        z_peak = zp,
        lamb = λ
    )
end

function bundle_from_hdf5(
        intrinsic_site_order::Vector{String},
        proposal_samples::Dict{String, Vector{Float64}}
)::FullBNSSamplesSoA
    intrinsic_site_order == FULL_BNS_INTRINSIC_ORDER || throw(
        ArgumentError(
        "unsupported intrinsic_site_order $(repr(intrinsic_site_order)); " *
        "only full BNS is supported: $(repr(FULL_BNS_INTRINSIC_ORDER))",
    ),
    )
    for key in FULL_BNS_INTRINSIC_ORDER
        haskey(proposal_samples, key) ||
            throw(ArgumentError("proposal_samples must include $(key) for full BNS layout"))
    end
    return (
        mass = stack_source_masses(
            proposal_samples["mass_1_source"],
            proposal_samples["mass_2_source"]
        ),
        redshift = copy(proposal_samples["redshift"]),
        chi_1 = copy(proposal_samples["chi_1"]),
        chi_2 = copy(proposal_samples["chi_2"]),
        lambda_1 = copy(proposal_samples["lambda_1"]),
        lambda_2 = copy(proposal_samples["lambda_2"])
    )
end

"""
    load_cache(path, detectors) -> ImportanceSamplingProblem

Read an HDF5 importance cache and return an [`ImportanceSamplingProblem`](@ref).

Root attributes **`command`** and **`git_revision`** (see [`IMPORTANCE_CACHE_COMMAND_ATTR`](@ref)
and [`IMPORTANCE_CACHE_GIT_REVISION_ATTR`](@ref)) record provenance. Required physics attributes
include `local_merger_rate`, `observation_time_sec`, and `observation_time_yr`. Optional
`redshift_integral_fiducial` is used when present; otherwise it is recomputed from fiducial
hyperparameters and [`RedshiftPriorSpec`](@ref).

The cache must contain dataset `cached_flux` (per-frequency flux before the fiducial
`(D_L/D_gw)^2` factor), which is converted to internal `cached_flux_over_dgw2` on load.
Two-dimensional datasets `cached_flux` and `proposal_intrinsic_vector` use HDF5 extent
`(n_columns, n_samples)` and are normalized to `(n_samples, n_columns)` in memory.
Datasets `covariance`, `effective_psd`, and `sgwb_scale` must **not** be present; pass `detectors` as a vector
of at least two [`Detector`](@ref) values so [`effective_psd`](@ref) and `sgwb_scale` are built from tabulated PSDs and
overlap reduction functions.

Dataset `proposal_log_prob` may be omitted (reconstructed from samples, prior spec, and
fiducial population parameters). Dataset `dgw_fid_sq` may be omitted (reconstructed from
redshifts and fiducial cosmology). Population scalars for reconstruction may appear under
`hyperparameters` and/or `redshift_prior_spec`; if both define the same key, values must match.

The `proposal_samples` group must carry string attribute [`PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR`](@ref)
with value [`PROPOSAL_SAMPLES_SOURCE_TYPE_BNS`](@ref).

Dataset `fiducial_spectral_density` may be present in the file for provenance but is **ignored on load**:
[`load_cache`](@ref) always fills [`ObservationConfig`](@ref).`fiducial_spectral_density` by calling
[`fiducial_spectral_density`](@ref) on the assembled problem so the default likelihood data match the
current Julia forward model.

Equivalent in-memory problems can be built with [`importance_sampling_problem`](@ref).
"""
function load_cache(
        path::AbstractString,
        detectors::AbstractVector{D}
)::ImportanceSamplingProblem where {D <: Detector}
    isempty(detectors) && throw(ArgumentError("load_cache: detectors must be non-empty"))
    length(detectors) < 2 && throw(
        ArgumentError(
        "load_cache: at least two detectors are required to build effective_psd and sgwb_scale",
    ),
    )
    return h5open(path, "r") do file
        attrs = attributes(file)
        _read_nonempty_string_attr(attrs, IMPORTANCE_CACHE_COMMAND_ATTR)
        _read_nonempty_string_attr(attrs, IMPORTANCE_CACHE_GIT_REVISION_ATTR)

        (haskey(file, "covariance") ||
         haskey(file, "effective_psd") ||
         haskey(file, "sgwb_scale")) && throw(
            ArgumentError(
            "cache must not contain covariance, effective_psd, or sgwb_scale datasets; " *
            "pass detectors to reconstruct them from tabulated PSDs and ORFs",
        ),
        )
        haskey(file, "cached_flux_over_dgw2") && throw(
            ArgumentError(
            "cache must not contain cached_flux_over_dgw2; use dataset cached_flux instead",
        ),
        )

        intrinsic_site_order = _read_string_vector(_require_child(file, "intrinsic_site_order"))

        proposal_samples = Dict{String, Vector{Float64}}()
        proposal_samples_group = _require_child(file, "proposal_samples")
        _validate_proposal_samples_source_type(proposal_samples_group)
        proposal_samples["redshift"] = _read_float_vector(
            _require_child(proposal_samples_group, "redshift"),
            "proposal_samples/redshift"
        )
        n_samples = length(proposal_samples["redshift"])
        proposal_intrinsic_vector = _read_hdf5_col_sample_matrix(
            _require_child(file, "proposal_intrinsic_vector"),
            "proposal_intrinsic_vector",
            n_samples,
            length(intrinsic_site_order)
        )

        frequencies = _read_float_vector(_require_child(file, "frequencies"), "frequencies")
        in_band_mask = _read_in_band_mask(_require_child(file, "in_band_mask"))
        # Placeholder only; replaced after `importance_sampling_problem` by `fiducial_spectral_density(p)`.
        # Any HDF5 `fiducial_spectral_density` dataset is ignored so caches cannot go stale vs Julia code.
        fiducial_spectral_density_vec = zeros(Float64, length(frequencies))

        obs_sec = Float64(_read_attr(attrs, "observation_time_sec"))
        obs_yr = Float64(_read_attr(attrs, "observation_time_yr"))
        det_vec = Vector{Detector}(collect(detectors))
        observation0 = build_observation_config(
            collect(Float64, frequencies),
            det_vec,
            in_band_mask,
            collect(Float64, fiducial_spectral_density_vec),
            obs_sec,
            obs_yr
        )
        effective_psd = observation0.effective_psd
        sgwb_scale = observation0.sgwb_scale

        for key in intrinsic_site_order
            key == "redshift" && continue
            proposal_samples[key] = _read_float_vector(
                _require_child(proposal_samples_group, key),
                "proposal_samples/$(key)"
            )
        end

        hyper_group = _require_child(file, "hyperparameters")
        spec_group = _require_child(file, "redshift_prior_spec")
        redshift_prior_spec = _read_redshift_prior_spec(file)
        fiducial_parameters = _read_proposal_fiducial_parameters(
            hyper_group,
            spec_group,
            redshift_prior_spec.family
        )

        haskey(proposal_samples, "redshift") ||
            throw(ArgumentError("proposal_samples must include a redshift entry"))
        samples_bundle = bundle_from_hdf5(intrinsic_site_order, proposal_samples)

        proposal_log_prob = if haskey(file, "proposal_log_prob")
            _read_float_vector(_require_child(file, "proposal_log_prob"), "proposal_log_prob")
        else
            reconstruct_proposal_log_prob(
                samples_bundle,
                redshift_prior_spec,
                fiducial_parameters
            )
        end

        haskey(file, "cached_flux") ||
            throw(ArgumentError("missing required HDF5 dataset: cached_flux"))
        cached_flux_over_dgw2 = reconstruct_cached_flux_over_dgw2(
            _read_hdf5_freq_sample_matrix(
                _require_child(file, "cached_flux"),
                "cached_flux",
                n_samples,
                length(frequencies)
            ),
            proposal_samples["redshift"],
            fiducial_parameters
        )

        dgw_fid_sq = if haskey(file, "dgw_fid_sq")
            _read_float_vector(_require_child(file, "dgw_fid_sq"), "dgw_fid_sq")
        else
            reconstruct_dgw_fid_sq(proposal_samples["redshift"], fiducial_parameters)
        end

        length(intrinsic_site_order) == size(proposal_intrinsic_vector, 2) || throw(
            ArgumentError(
            "proposal_intrinsic_vector column count must match intrinsic_site_order length",
        ),
        )
        size(proposal_intrinsic_vector, 1) == n_samples || throw(
            ArgumentError(
            "proposal_intrinsic_vector row count must match redshift / sample count",
        ),
        )
        size(cached_flux_over_dgw2, 2) == n_samples || throw(
            ArgumentError(
            "cached flux matrix column count must match redshift / sample count",
        ),
        )
        length(dgw_fid_sq) == n_samples ||
            throw(ArgumentError("dgw_fid_sq length must match redshift / sample count"))
        length(proposal_log_prob) == n_samples || throw(
            ArgumentError("proposal_log_prob length must match redshift / sample count"),
        )
        for key in intrinsic_site_order
            length(proposal_samples[key]) == n_samples || throw(
                ArgumentError(
                "proposal_samples/$(key) length must match redshift / sample count",
            ),
            )
        end

        n_frequencies = length(frequencies)
        size(cached_flux_over_dgw2, 1) == n_frequencies ||
            throw(ArgumentError("cached flux row count must match frequencies length"))
        length(effective_psd) == n_frequencies ||
            throw(ArgumentError("effective_psd length must match frequencies length"))
        length(sgwb_scale) == n_frequencies ||
            throw(ArgumentError("sgwb_scale length must match frequencies length"))
        length(in_band_mask) == n_frequencies ||
            throw(ArgumentError("in_band_mask length must match frequencies length"))
        length(fiducial_spectral_density_vec) == n_frequencies || throw(
            ArgumentError("fiducial_spectral_density length must match frequencies length"),
        )

        proposal = ProposalData(
            intrinsic_site_order,
            samples_bundle,
            proposal_log_prob,
            proposal_intrinsic_vector,
            cached_flux_over_dgw2,
            dgw_fid_sq
        )

        observation = ObservationConfig(
            frequencies,
            effective_psd,
            sgwb_scale,
            in_band_mask,
            fiducial_spectral_density_vec,
            obs_sec,
            obs_yr
        )

        redshift_integral_fiducial = if haskey(attrs, "redshift_integral_fiducial")
            Float64(_read_attr(attrs, "redshift_integral_fiducial"))
        else
            try
                fiducial_redshift_integral(fiducial_parameters, redshift_prior_spec)
            catch err
                throw(
                    ArgumentError(
                    "cache omits redshift_integral_fiducial but recomputation failed; " *
                    "ensure population keys are present in hyperparameters or redshift_prior_spec " *
                    "(e.g. gamma, kappa, z_peak for Madau–Dickinson, or lamb for power-law). " *
                    "Underlying error: " *
                    sprint(showerror, err),
                ),
                )
            end
        end

        p = importance_sampling_problem(
            proposal,
            observation,
            redshift_prior_spec,
            Float64(_read_attr(attrs, "local_merger_rate")),
            redshift_integral_fiducial,
            fiducial_parameters
        )
        fs = try
            fiducial_spectral_density(p)
        catch err
            throw(
                ArgumentError(
                "fiducial_spectral_density recomputation on load failed; " *
                "ensure population keys are present for the redshift prior. " *
                "Underlying error: " *
                sprint(showerror, err),
            ),
            )
        end
        length(fs) == n_frequencies || throw(
            ArgumentError(
            "fiducial_spectral_density recomputation returned length $(length(fs)), expected $n_frequencies",
        ),
        )
        observation2 = ObservationConfig(
            p.observation.frequencies,
            p.observation.effective_psd,
            p.observation.sgwb_scale,
            p.observation.in_band_mask,
            fs,
            p.observation.observation_time_sec,
            p.observation.observation_time_yr
        )
        p = importance_sampling_problem(
            p.proposal,
            observation2,
            p.redshift_prior_spec,
            p.local_merger_rate,
            p.redshift_integral_fiducial,
            p.fiducial_parameters
        )
        return p
    end
end
