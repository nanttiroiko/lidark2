"""Tests for lidark.maps.api: create_map validation and dispatch."""

from pathlib import Path
from typing import Any

import pytest

from lidark.config import DEFAULT_EPSG
from lidark.maps import create_map
from lidark.maps.api import MAP_TYPES


def test_map_types() -> None:
    assert MAP_TYPES == ("DTM", "TPI")


def test_create_map_rejects_unknown_type(tmp_path: Path) -> None:
    # Fixes B1: validation happens before any processing, with one clear error.
    for bad in ("MDHS", "dtm", "dem", ""):
        with pytest.raises(ValueError, match="map_type must be one of"):
            create_map(tmp_path, "Naljanka", bad)


def test_create_map_dispatches_dtm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_generate_dtm(layout, area, *, pixel_size, buffer, cores):
        calls["dtm"] = (layout, area, pixel_size, buffer, cores)
        return [Path("tile_DTM.tif")]

    import lidark.maps.dtm as dtm_module

    monkeypatch.setattr(dtm_module, "generate_dtm", fake_generate_dtm)
    result = create_map(
        tmp_path, "Naljanka", "DTM", buffer=50.0, pixel_size=0.5, cores=4, epsg=DEFAULT_EPSG
    )
    assert result == [Path("tile_DTM.tif")]
    layout, area, pixel_size, buffer, cores = calls["dtm"]
    assert layout.project_folder == tmp_path
    assert layout.epsg == DEFAULT_EPSG
    assert area == "Naljanka"
    assert (pixel_size, buffer, cores) == (0.5, 50.0, 4)


def test_create_map_dtm_buffer_defaults_to_2m(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    def fake_generate_dtm(layout, area, *, pixel_size, buffer, cores):
        calls["buffer"] = buffer
        return []

    import lidark.maps.dtm as dtm_module

    monkeypatch.setattr(dtm_module, "generate_dtm", fake_generate_dtm)
    # No --buffer: the DTM default of 2.0 m is selected.
    create_map(tmp_path, "Naljanka", "DTM")
    assert calls["buffer"] == 2.0


def test_create_map_tpi_radius_defaults_to_30m(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    def fake_create_tpi(layout, area, *, radius, gpu, cores, **kwargs: Any):
        calls["radius"] = radius
        return []

    import lidark.maps.tpi as tpi_module

    monkeypatch.setattr(tpi_module, "create_tpi", fake_create_tpi)
    create_map(tmp_path, "Naljanka", "TPI")
    assert calls["radius"] == 30.0


def test_create_map_dispatches_tpi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_create_tpi(layout, area, *, radius, gpu, cores, **kwargs: Any):
        calls["tpi"] = (layout, area, radius, gpu, cores)
        return [Path("tile_TPI.tif")]

    import lidark.maps.tpi as tpi_module

    monkeypatch.setattr(tpi_module, "create_tpi", fake_create_tpi)
    result = create_map(tmp_path, "Naljanka", "TPI", buffer=20.0, gpu=True, cores=2)
    assert result == [Path("tile_TPI.tif")]
    layout, area, radius, gpu, cores = calls["tpi"]
    assert area == "Naljanka"
    assert (radius, gpu, cores) == (20.0, True, 2)
