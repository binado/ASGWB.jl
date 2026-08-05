"""Generate the cross-language catalog parity fixture for Julia tests.

Writes a small ``waveform_catalog`` v1 file together with the polarization power
that the Python ``astrogwb`` package derives from it. The Julia test in
``AstroSGWB/test/test_io.jl`` loads the same ``.h5`` with ``load_catalog`` and
asserts its ``fluxes`` match the ``.npz`` reference -- i.e. that both languages
reduce one catalog file to the same numbers.

This imports ``astrogwb`` deliberately: inlining the reduction here would only
test this script against itself. Run it with the ``astrogwb`` checkout's
interpreter, from the AstroSGWB.jl repo root::

    ../astrogwb/.venv/bin/python3 scripts/generate_catalog_parity_fixture.py

Writes (both are gitignored, and the Julia test skips when they are absent):

- ``AstroSGWB/test/fixtures/catalog_parity_reference.h5`` -- the catalog itself.
- ``AstroSGWB/test/fixtures/catalog_parity_reference.npz`` -- arrays
  ``fluxes`` (``(nfreq, nsamples)``), ``frequencies``, and ``in_band_mask``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pluscross
from astrogwb.waveform.polarization_power import polarization_power

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[1] / "AstroSGWB" / "test" / "fixtures"

NSAMPLES = 8
NFREQ = 16
SAMPLING_FREQUENCY = 64.0
MINIMUM_FREQUENCY = 4.0
MAXIMUM_FREQUENCY = 24.0
REFERENCE_FREQUENCY = 8.0


def main() -> None:
    rng = np.random.default_rng(20260804)
    frequencies = np.arange(NFREQ, dtype=np.float64) * 2.0

    # Amplitudes spanning many orders of magnitude, so the comparison exercises
    # the reduction across the dynamic range of a real catalog rather than
    # around unity. Both polarizations carry power and both parts are non-zero.
    def draw() -> np.ndarray:
        scale = 10.0 ** rng.uniform(-25.0, -20.0, size=(NSAMPLES, NFREQ))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(NSAMPLES, NFREQ))
        return scale * np.exp(1j * phase)

    catalog = pluscross.WaveformCatalog(
        frequencies=frequencies,
        plus=draw(),
        cross=draw(),
        source_parameters={
            "redshift": rng.uniform(0.01, 2.0, size=NSAMPLES),
            "luminosity_distance": rng.uniform(100.0, 5000.0, size=NSAMPLES),
            "inclination": np.zeros(NSAMPLES),
        },
        approximant="IMRPhenomXAS_NRTidalv3",
        minimum_frequency=MINIMUM_FREQUENCY,
        maximum_frequency=MAXIMUM_FREQUENCY,
        reference_frequency=REFERENCE_FREQUENCY,
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    h5_path = FIXTURE_DIR / "catalog_parity_reference.h5"
    npz_path = FIXTURE_DIR / "catalog_parity_reference.npz"

    pluscross.save_catalog(str(h5_path), catalog)

    # Reduce the file as written, not the in-memory object, so any IO-side
    # rounding is part of what the Julia side is compared against.
    reloaded = pluscross.load_catalog(str(h5_path))
    fluxes = polarization_power(reloaded)
    in_band_mask = (reloaded.frequencies >= reloaded.minimum_frequency) & (
        reloaded.frequencies <= reloaded.maximum_frequency
    )

    np.savez(
        npz_path,
        fluxes=fluxes,
        frequencies=reloaded.frequencies,
        in_band_mask=in_band_mask,
    )
    print(f"wrote {h5_path}")
    print(f"wrote {npz_path} (fluxes shape {fluxes.shape})")


if __name__ == "__main__":
    main()
