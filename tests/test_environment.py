"""Verifiche delle capacita' runtime dell'ambiente.

Non testano codice del progetto ma le assunzioni su cui poggia: il backend GRIB
va registrato in xarray e i binari ECMWF devono essere raggiungibili, cosa che
dipende da wheel e librerie di sistema e cambia tra host e immagine Docker.
Servono quindi anche come smoke test dell'immagine.
"""

from __future__ import annotations

import importlib

import pytest

RUNTIME_PACKAGES = [
    "numpy",
    "pandas",
    "polars",
    "xarray",
    "zarr",
    "cfgrib",
    "eccodes",
    "cdsapi",
    "pydantic",
    "yaml",
    "dask",
    "scipy",
    "torch",
    "matplotlib",
]


@pytest.mark.parametrize("package", RUNTIME_PACKAGES)
def test_pacchetto_importabile(package: str) -> None:
    importlib.import_module(package)


def test_backend_grib_registrato_in_xarray() -> None:
    """Senza il backend cfgrib l'apertura dei GRIB scaricati fallirebbe a runtime."""
    import xarray as xr

    assert "cfgrib" in xr.backends.list_engines()


def test_binari_eccodes_raggiungibili() -> None:
    """cfgrib e' solo un wrapper: serve la libreria ECMWF compilata."""
    import eccodes

    version = eccodes.codes_get_api_version()
    assert isinstance(version, str)
    assert version.split(".")[0].isdigit()


def test_torch_funziona_su_cpu() -> None:
    import torch

    tensor = torch.ones(4, 3, dtype=torch.float32)
    assert tensor.device.type == "cpu"
    assert float((tensor * 2).sum()) == pytest.approx(24.0)


def test_torch_e_build_cpu() -> None:
    """L'immagine deve restare leggera: i wheel CUDA sono esclusi via indice dedicato."""
    import torch

    assert not torch.version.cuda, f"build CUDA inattesa: {torch.version.cuda}"


def test_conv2d_supporta_padding_riflesso() -> None:
    """Il modello usa padding riflesso ai bordi del dominio, non zeri."""
    import torch
    from torch import nn

    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, padding_mode="reflect")
    out = layer(torch.zeros(1, 2, 8, 8))
    assert out.shape == (1, 3, 8, 8)


def test_zarr_e_versione_tre() -> None:
    """L'API dei codec differisce tra zarr 2 e 3: il codice assume la 3."""
    import zarr

    assert int(zarr.__version__.split(".")[0]) >= 3


def test_polars_scrive_e_rilegge_parquet(tmp_path) -> None:
    """Parquet e' il formato del layer tabellare: verifica il round-trip completo."""
    import polars as pl

    frame = pl.DataFrame({"slot_index": [0, 1, 2], "value": [1.5, 2.5, 3.5]})
    target = tmp_path / "round_trip.parquet"
    frame.write_parquet(target)
    assert pl.read_parquet(target).equals(frame)
