# HDF5 importance cache specification

This document describes the HDF5 layout consumed by `load_cache` in **ASGWB.jl** (see [`src/io.jl`](src/io.jl)). Files that do not match this layout are rejected.

## Root attributes

All attributes live on the HDF5 **root group** (`/`).

| Attribute | Type | Required | Meaning |
| --------- | ---- | -------- | ------- |
| `command` | string | yes | Full shell command used to generate the cache (provenance). |
| `git_revision` | string | yes | Git object id of the generator codebase (provenance). |
| `local_merger_rate` | real | yes | Local merger rate scale passed to `importance_sampling_problem`. |
| `observation_time_sec` | real | yes | Observation duration in seconds. |
| `observation_time_yr` | real | yes | Observation duration in years. |
| `redshift_integral_fiducial` | real | no | When omitted, set to `fiducial_redshift_integral` from fiducial population parameters and `RedshiftPriorSpec`. |

## Root groups (required)

### `hyperparameters`

Scalar datasets (0-dimensional or single-element), read as `Float64`.

**Cosmology / propagation (required):** `H0`, `Omega_m`, `chi0`, `chin`.

**Population (optional on disk):** `gamma`, `kappa`, `z_peak` for Madau–Dickinson; `lamb` for power-law. The same keys may also appear under `redshift_prior_spec` (see below). If a key appears in both places, values must match exactly.

Unknown dataset names in this group are rejected.

### `redshift_prior_spec`

| Entry | Required | Type | Notes |
| ----- | -------- | ---- | ----- |
| `family` | yes | string | `madau_dickinson` or `power_law` (snake_case; `parse_redshift_prior_family` in `src/types.jl`). |
| `z_min`, `z_max` | yes | real | |
| `num_interp` | yes | integer | Grid size; may be stored as `I64` or `I32`. |
| `time_delay_model` | no | string | If absent or empty, loads as `nothing`. |

**Optional duplicate population scalars:** `gamma`, `kappa`, `z_peak` (Madau–Dickinson) or `lamb` (power-law). Used when reconstructing omitted `proposal_log_prob` / fiducial spectrum if the corresponding entries are missing from `hyperparameters`.

Extra datasets in this group are allowed (ignored except for population merge and consistency checks above).

### `proposal_samples`

One **1D float dataset per name** in `intrinsic_site_order`. Full BNS order is:

`mass_1_source`, `mass_2_source`, `redshift`, `chi_1`, `chi_2`, `lambda_1`, `lambda_2`.

**Group attribute (required):** `source_type` (string) must be exactly `BNS` (see `PROPOSAL_SAMPLES_SOURCE_TYPE_BNS`).

## Root datasets

| Dataset | Required | Notes |
| ------- | -------- | ----- |
| `intrinsic_site_order` | yes | 1D array of strings; must be the full BNS order above. |
| `proposal_intrinsic_vector` | yes | 2D float; **HDF5 extent** `(n_intrinsic, n_samples)` = `(n_cols, n_samples)` (`h5dump` lists the intrinsic index first). The loader normalizes to `(n_samples, n_intrinsic)` in Julia (see “2D array storage”). |
| `frequencies` | yes | 1D float, non-empty. |
| `in_band_mask` | yes | 1D mask; may be bool, integer, or HDF5 enum (`FALSE`/`TRUE`). |
| `cached_flux` | yes | 2D float; **HDF5 extent** `(n_f, n_samples)` = `(n_cols, n_samples)`; flux **before** multiplying by `(D_L/D_gw)^2` under fiducial `hyperparameters`. Normalized to `(n_samples, n_f)` in Julia. |
| `fiducial_spectral_density` | no | 1D float, length `n_f`. If omitted, recomputed via `fiducial_spectral_density` (requires population keys as for omitted `proposal_log_prob`). |

**Must not be present:** `covariance`, `sgwb_scale`, `cached_flux_over_dgw2`.

**Optional:** `proposal_log_prob` (1D float, length `n_samples`); `dgw_fid_sq` (1D float, length `n_samples`). If omitted, they are reconstructed (`src/cache.jl`).

## `load_cache` API

```julia
load_cache(path::AbstractString, detectors::AbstractVector{<:Detector}) -> ImportanceSamplingProblem
```

Pass **at least two** detectors. Covariance and `sgwb_scale` are always built from tabulated PSDs and overlap reduction functions for the given detector network.

## 2D array storage

`proposal_intrinsic_vector` and `cached_flux` use a single **on-disk layout**: HDF5 dataspace extent **`(n_columns, n_samples)`** — for `cached_flux`, `n_columns = n_f`; for `proposal_intrinsic_vector`, `n_columns = n_intrinsic` (same order as `intrinsic_site_order`). Downstream code always sees **`(n_samples, n_columns)`** (rows = samples). `HDF5.read` may return either `(n_columns, n_samples)` or `(n_samples, n_columns)` for that extent; the loader applies `permutedims` only in the former case. If `n_f == n_samples` (a square dataset), this check is ambiguous—use non-square caches or ensure your writer matches one of the two shapes above.

## Reference

Authoritative behavior is `load_cache` and helpers in `src/io.jl`, `src/cache.jl`, and tests under `test/test_io.jl`. To refresh committed binary fixtures from an older tree, see `contrib/upgrade_hdf5_importance_caches.jl`.
