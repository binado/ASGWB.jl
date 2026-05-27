#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "h5py>=3.0",
#   "astropy>=6.0",
#   "bilby>=2.0",
#   "lalsuite",
# ]
# ///
"""
Standalone raw waveform-power accumulator for CBC injection catalogs.
No internal package dependencies; requires only: numpy, astropy, pandas, h5py, and bilby.

Processes a large whitespace-separated injection catalog in parallel chunks,
accumulates the raw sum of |h_plus|^2 + |h_cross|^2 over catalog rows, and
writes a compact HDF5 file. The output dataset is named ``spectral_density``
for compatibility with early standalone consumers, but no astrophysical rate,
bin-width, or event-count normalization is applied.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from functools import cached_property
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("generate_waveforms")

# Types
SourceType = Literal["BBH", "BNS"]


# ==============================================================================
# 1. Frequency Grid Management
# ==============================================================================
@dataclass(frozen=True)
class FrequencyGrid:
    duration: float
    sampling_frequency: float
    reference_frequency: float
    minimum_frequency: float = 10.0
    maximum_frequency: float | None = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.sampling_frequency <= 0:
            raise ValueError(f"sampling_frequency must be positive, got {self.sampling_frequency}")

        nyquist = self.sampling_frequency / 2.0
        resolved_max = nyquist if self.maximum_frequency is None else self.maximum_frequency

        if self.minimum_frequency < 0:
            raise ValueError(f"minimum_frequency must be non-negative, got {self.minimum_frequency}")
        if self.minimum_frequency >= resolved_max:
            raise ValueError("minimum_frequency must be less than maximum_frequency")
        if resolved_max > nyquist:
            raise ValueError(f"maximum_frequency must be <= Nyquist ({nyquist})")

        object.__setattr__(self, "maximum_frequency", resolved_max)

    @property
    def frequencies(self) -> np.ndarray:
        nsamples = int(np.rint(self.duration * self.sampling_frequency))
        nfrequencies = nsamples // 2 + 1
        return np.linspace(0, self.sampling_frequency / 2.0, nfrequencies, dtype=np.float64)

    @property
    def in_band_mask(self) -> np.ndarray:
        return (self.frequencies >= self.minimum_frequency) & (self.frequencies <= self.maximum_frequency)


# ==============================================================================
# 2. Cosmology Handlers
# ==============================================================================
def setup_cosmology(parameters: dict[str, float]) -> Any:
    """Setup astropy cosmology with given parameters (or Planck18 defaults)."""
    from astropy.cosmology import FlatLambdaCDM, Planck18
    merged = Planck18.parameters.copy()
    if "H0" in parameters:
        merged["H0"] = parameters["H0"]
    if "Omega_m" in parameters:
        merged["Om0"] = parameters["Omega_m"]
    return FlatLambdaCDM(**merged)


def apply_bilby_cosmology(cosmology: Any) -> None:
    """Configure active cosmology within bilby."""
    import importlib
    try:
        bilby_cosmology = importlib.import_module("bilby.gw.cosmology")
    except ImportError:
        bilby_cosmology = importlib.import_module("bilby.core.utils.cosmology")
    bilby_cosmology.set_cosmology(cosmology)


# ==============================================================================
# 3. Standalone Waveform Backend
# ==============================================================================
class WaveformGenerator:
    def __init__(
        self,
        approximant: str,
        grid: FrequencyGrid,
        source_type: SourceType = "BNS",
        cosmology: Any | None = None,
    ) -> None:
        self.approximant = approximant
        self.grid = grid
        self.source_type = source_type
        self.cosmology = cosmology

    @cached_property
    def bilby_generator(self) -> Any:
        # Defer imports to keep CLI responsive
        from bilby.gw.conversion import (
            convert_to_lal_binary_black_hole_parameters,
            convert_to_lal_binary_neutron_star_parameters,
        )
        from bilby.gw.source import (
            gwsignal_binary_black_hole,
            lal_binary_black_hole,
            lal_binary_neutron_star,
        )
        from bilby.gw.waveform_generator import WaveformGenerator as BilbyWaveformGenerator

        if self.cosmology is not None:
            apply_bilby_cosmology(self.cosmology)

        if self.source_type == "BBH":
            source_model = (
                gwsignal_binary_black_hole
                if self.approximant in {"SEOBNRv5HM", "SEOBNRv5PHM"}
                else lal_binary_black_hole
            )
            parameter_conversion = convert_to_lal_binary_black_hole_parameters
        elif self.source_type == "BNS":
            source_model = lal_binary_neutron_star
            parameter_conversion = convert_to_lal_binary_neutron_star_parameters
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}")

        waveform_arguments = {
            "waveform_approximant": self.approximant,
            "reference_frequency": self.grid.reference_frequency,
            "minimum_frequency": self.grid.minimum_frequency,
            "maximum_frequency": self.grid.maximum_frequency,
        }

        return BilbyWaveformGenerator(
            parameters=None,
            frequency_domain_source_model=source_model,
            duration=self.grid.duration,
            sampling_frequency=self.grid.sampling_frequency,
            parameter_conversion=parameter_conversion,
            waveform_arguments=waveform_arguments,
        )

    def generate(self, parameters: dict[str, float]) -> dict[str, np.ndarray]:
        """Generates plus and cross polarizations for a given parameter set."""
        strain = self.bilby_generator.frequency_domain_strain(parameters)
        if strain is None or "plus" not in strain or "cross" not in strain:
            raise RuntimeError("Waveform generation returned empty strain values.")
        return {"plus": strain["plus"], "cross": strain["cross"]}


# ==============================================================================
# 4. Multiprocessing Structures & Helpers
# ==============================================================================
@dataclass(frozen=True)
class WaveformGeneratorConfig:
    waveform_approximant: str
    reference_frequency: float
    sampling_frequency: float
    minimum_frequency: float
    maximum_frequency: float | None
    duration: float
    source_type: SourceType
    cosmology_params: dict[str, float]


@dataclass(frozen=True)
class ChunkPartialSum:
    partial_sum: np.ndarray
    processed: int
    chunk_start: int
    chunk_end: int


# Worker process global state
_WORKER_WAVEFORM_GENERATOR: WaveformGenerator | None = None


def _init_worker(config: WaveformGeneratorConfig) -> None:
    global _WORKER_WAVEFORM_GENERATOR
    cosmology = setup_cosmology(config.cosmology_params) if config.cosmology_params else None

    grid = FrequencyGrid(
        duration=config.duration,
        sampling_frequency=config.sampling_frequency,
        reference_frequency=config.reference_frequency,
        minimum_frequency=config.minimum_frequency,
        maximum_frequency=config.maximum_frequency,
    )

    _WORKER_WAVEFORM_GENERATOR = WaveformGenerator(
        approximant=config.waveform_approximant,
        grid=grid,
        source_type=config.source_type,
        cosmology=cosmology,
    )


def _validated_numeric_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    numeric = chunk.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    bad_positions = np.argwhere(~np.isfinite(values))

    if bad_positions.size == 0:
        return numeric

    examples = []
    for row_idx, col_idx in bad_positions[:5]:
        examples.append(f"row {chunk.index[row_idx]}, column {chunk.columns[col_idx]!r}")
    suffix = "" if len(bad_positions) <= 5 else f", plus {len(bad_positions) - 5} more"
    raise ValueError(
        "Injection catalog contains nonnumeric or nonfinite values at "
        + "; ".join(examples)
        + suffix
    )


def _compute_partial_sum_for_chunk(chunk: pd.DataFrame, generator: WaveformGenerator) -> ChunkPartialSum | None:
    if chunk.empty:
        return None

    chunk_start = int(chunk.index[0])
    chunk_end = int(chunk.index[-1])
    logger.info("Processing injections %d-%d", chunk_start, chunk_end)

    chunk = _validated_numeric_chunk(chunk)
    columns = chunk.columns.tolist()
    partial_sum: np.ndarray | None = None
    processed = 0

    for row in chunk.itertuples():
        injection_parameters = {
            column: float(value) for column, value in zip(columns, row[1:], strict=True)
        }

        # Apply cosmology scaling inside the parameters if needed
        if generator.cosmology is not None:
            if "redshift" in injection_parameters and "luminosity_distance" not in injection_parameters:
                injection_parameters["luminosity_distance"] = generator.cosmology.luminosity_distance(
                    injection_parameters["redshift"]
                ).value

        # Calculate polarizations (plus & cross) with fixed orientation/sky parameters
        params_copy = injection_parameters.copy()
        params_copy.update({
            "phase": 0.0,
            "ra": 0.0,
            "dec": 0.0,
            "geocent_time": 0.0,
            "psi": 0.0,
            "theta_jn": 0.0,
        })

        polarizations = generator.generate(params_copy)

        # Raw unnormalized sum of squared polarizations.
        contribution = (
            np.square(polarizations["plus"].real) + np.square(polarizations["plus"].imag) +
            np.square(polarizations["cross"].real) + np.square(polarizations["cross"].imag)
        )
        contribution = np.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0)

        if partial_sum is None:
            partial_sum = contribution.copy()
        else:
            partial_sum += contribution
        processed += 1

    assert partial_sum is not None
    return ChunkPartialSum(
        partial_sum=partial_sum,
        processed=processed,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
    )


def _compute_partial_sum_for_chunk_worker(chunk: pd.DataFrame) -> ChunkPartialSum | None:
    if _WORKER_WAVEFORM_GENERATOR is None:
        raise RuntimeError("Waveform generator not initialized in worker process.")
    return _compute_partial_sum_for_chunk(chunk, _WORKER_WAVEFORM_GENERATOR)


def reindex_chunks(chunks: Any, start_index: int) -> Any:
    next_index = start_index
    for chunk in chunks:
        stop_index = next_index + len(chunk)
        chunk.index = pd.RangeIndex(start=next_index, stop=stop_index)
        yield chunk
        next_index = stop_index


def _accumulate_chunk_result(sum_abs_sq: np.ndarray, chunk_result: ChunkPartialSum | None) -> tuple[int, int]:
    if chunk_result is None:
        return 0, 0

    if chunk_result.partial_sum.shape != sum_abs_sq.shape:
        raise ValueError(
            f"Frequency grid mismatch in chunk {chunk_result.chunk_start}-{chunk_result.chunk_end}: "
            f"{chunk_result.partial_sum.shape} vs {sum_abs_sq.shape}"
        )

    logger.info(
        "Accumulated %d events from chunk %d-%d",
        chunk_result.processed,
        chunk_result.chunk_start,
        chunk_result.chunk_end,
    )
    sum_abs_sq += chunk_result.partial_sum
    return chunk_result.processed, 1


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")
    if args.nworkers <= 0:
        parser.error("--nworkers must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.batch is not None and args.batch < 0:
        parser.error("--batch must be non-negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone CBC raw waveform-power streaming calculator")

    # Inputs & Outputs
    parser.add_argument("--injection-file", type=str, required=True, help="Path to space-separated CSV BNS/BBH injection catalog.")
    parser.add_argument("--output", type=str, required=True, help="Output path for the generated raw waveform-power HDF5 file (.h5).")

    # Grid Config
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--sampling-rate", type=float, default=2048.0)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=1024.0)
    parser.add_argument("--f-ref", type=float, default=50.0)

    # Model Config
    parser.add_argument("--approximant", type=str, default="IMRPhenomPV2_NRTidalv2")
    parser.add_argument("--source-type", type=str, choices=["BNS", "BBH"], default="BNS")
    parser.add_argument("--H0", type=float, help="Custom H0 for flat cosmology setting.")
    parser.add_argument("--omega-m", type=float, help="Custom Omega_m for flat cosmology setting.")

    # Batch Processing Config
    parser.add_argument("--chunksize", type=int, default=1000, help="Chunksize for streaming the injection catalog.")
    parser.add_argument("--nworkers", type=int, default=1, help="Number of worker processes for parallel generation.")
    parser.add_argument("--offset", type=int, default=0, help="Skip leading rows in the injection catalog.")
    parser.add_argument("--batch", type=int, help="Total number of injection catalog rows to process.")

    args = parser.parse_args()
    _validate_args(parser, args)
    return args


# ==============================================================================
# CLI Entrypoint & Runner
# ==============================================================================
def main() -> None:
    args = parse_args()

    # Determine Cosmology
    cosmo_params = {}
    if args.H0 is not None:
        cosmo_params["H0"] = args.H0
    if args.omega_m is not None:
        cosmo_params["Omega_m"] = args.omega_m
    cosmology = setup_cosmology(cosmo_params) if cosmo_params else None

    # Resolve Grid
    grid = FrequencyGrid(
        duration=args.duration,
        sampling_frequency=args.sampling_rate,
        minimum_frequency=args.f_min,
        maximum_frequency=args.f_max,
        reference_frequency=args.f_ref,
    )

    injection_path = Path(args.injection_file)
    if not injection_path.exists():
        raise FileNotFoundError(f"Injection catalog not found: {injection_path}")

    # Initialize cumulative array
    frequency_axis = grid.frequencies
    sum_abs_sq = np.zeros(frequency_axis.shape[0], dtype=np.float64)

    # Configure pandas read parameters for chunk-streaming
    read_kwargs = {}
    if args.offset > 0:
        read_kwargs["skiprows"] = range(1, args.offset + 1)
    if args.batch is not None:
        read_kwargs["nrows"] = args.batch

    # Stream space-separated CSV chunks
    reader = pd.read_csv(
        injection_path,
        sep=r"\s+",
        header=0,
        comment="#",
        engine="c",
        iterator=True,
        chunksize=args.chunksize,
        **read_kwargs,
    )
    indexed_reader = reindex_chunks(reader, start_index=args.offset)

    config = WaveformGeneratorConfig(
        waveform_approximant=args.approximant,
        reference_frequency=args.f_ref,
        sampling_frequency=args.sampling_rate,
        minimum_frequency=args.f_min,
        maximum_frequency=args.f_max,
        duration=args.duration,
        source_type=args.source_type,
        cosmology_params=cosmo_params,
    )

    n_events_processed = 0
    n_chunks = 0

    # Processing loop
    if args.nworkers == 1:
        logger.info("Processing catalog in a single process...")
        local_generator = WaveformGenerator(
            approximant=args.approximant,
            grid=grid,
            source_type=args.source_type,
            cosmology=cosmology,
        )
        for chunk in indexed_reader:
            processed, chunk_count = _accumulate_chunk_result(
                sum_abs_sq=sum_abs_sq,
                chunk_result=_compute_partial_sum_for_chunk(chunk, local_generator),
            )
            n_events_processed += processed
            n_chunks += chunk_count
    else:
        logger.info("Processing catalog in parallel using Pool with %d workers...", args.nworkers)
        with Pool(
            processes=args.nworkers,
            initializer=_init_worker,
            initargs=(config,),
        ) as pool:
            for chunk_result in pool.imap_unordered(
                _compute_partial_sum_for_chunk_worker,
                indexed_reader,
                chunksize=1,
            ):
                processed, chunk_count = _accumulate_chunk_result(
                    sum_abs_sq=sum_abs_sq,
                    chunk_result=chunk_result,
                )
                n_events_processed += processed
                n_chunks += chunk_count

    # Write out to standard hdf5 spectral density format
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing raw unnormalized waveform-power sum to HDF5 file: %s", out_path)
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset(
            "spectral_density",
            data=sum_abs_sq,
            dtype=np.float64,
            compression="gzip",
        )
        # Write exact schema metadata attributes
        hf.attrs["grid_duration"] = grid.duration
        hf.attrs["grid_sampling_frequency"] = grid.sampling_frequency
        hf.attrs["grid_minimum_frequency"] = grid.minimum_frequency
        hf.attrs["grid_maximum_frequency"] = grid.maximum_frequency
        hf.attrs["grid_reference_frequency"] = grid.reference_frequency
        hf.attrs["n_events"] = n_events_processed
        hf.attrs["quantity"] = "raw_sum_abs_h_plus_cross_squared"
        hf.attrs["normalization"] = "none"

    logger.info(
        "Raw waveform-power sum successfully written to %s (%d events across %d chunks)",
        out_path,
        n_events_processed,
        n_chunks,
    )


if __name__ == "__main__":
    main()
