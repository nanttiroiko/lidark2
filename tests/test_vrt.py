"""Tests for lidark.io.vrt: gdalbuildvrt mosaics and windowed reads."""

from pathlib import Path

import numpy as np
import pytest
from rasterio.windows import Window
from tests.conftest import ORIGIN_X, ORIGIN_Y, PIXEL_SIZE, TILE_SHAPE, tile_array

from lidark.io import vrt
from lidark.io.tiles import build_tile_index
from lidark.io.vrt import build_vrt, open_vrt


def test_build_vrt_and_read_across_seam(dtm_tiles: list[Path], dtm_dir: Path) -> None:
    out = build_vrt(dtm_tiles, dtm_dir / "dtm.vrt")
    assert out.exists()

    with open_vrt(out) as mosaic:
        # Mosaic extent equals the union of the tile footprints.
        index = build_tile_index(dtm_dir)
        expected_bounds = (
            index.total_bounds[0],
            index.total_bounds[1],
            index.total_bounds[2],
            index.total_bounds[3],
        )
        actual = mosaic.bounds
        assert tuple(v == pytest.approx(e) for v, e in zip(actual, expected_bounds, strict=False))
        # Mosaic pixel size matches the tiles.
        assert mosaic.transform.a == PIXEL_SIZE

        # Windowed read of tile 0 equals its array.
        height, width = TILE_SHAPE
        win = Window(0, 0, width, height)
        tile0 = mosaic.read(1, window=win)
        assert np.array_equal(tile0, tile_array(0, 0))

        # Windowed read crossing the horizontal seam between tiles 0 and 1.
        seam = mosaic.read(1, window=Window(0, 0, 2 * width, height))
        assert np.array_equal(seam[:, :width], tile_array(0, 0))
        assert np.array_equal(seam[:, width:], tile_array(0, 1))


def test_open_vrt_point_lookup(dtm_tiles: list[Path], dtm_dir: Path) -> None:
    out = build_vrt(dtm_tiles, dtm_dir / "dtm.vrt")
    with open_vrt(out) as mosaic:
        # Point inside tile 3 (row 1, col 1).
        x = ORIGIN_X + 1.5 * TILE_SHAPE[1] * PIXEL_SIZE + 2.0
        y = ORIGIN_Y - 1.5 * TILE_SHAPE[0] * PIXEL_SIZE - 2.0
        row, col = mosaic.index(x, y)
        value = mosaic.read(1, window=Window(col, row, 1, 1))[0, 0]
        assert value == pytest.approx(
            tile_array(1, 1)[int(row) % TILE_SHAPE[0], int(col) % TILE_SHAPE[1]]
        )


def test_build_vrt_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        build_vrt([], tmp_path / "out.vrt")


def test_build_vrt_missing_binary_raises(
    dtm_tiles: list[Path], dtm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vrt,
        "_gdalbuildvrt_path",
        lambda: (_ for _ in ()).throw(FileNotFoundError("gdalbuildvrt not found")),
    )
    with pytest.raises(FileNotFoundError):
        build_vrt(dtm_tiles, dtm_dir / "x.vrt")


def test_resolver_finds_binary_in_conda_env() -> None:
    # The fallback (relative to the running interpreter) locates the conda
    # gdalbuildvrt even when PATH does not contain it.
    binary = vrt._gdalbuildvrt_path()
    assert Path(binary).exists()


def _fake_run_factory(captured: dict, returncode: int = 0) -> object:
    """A subprocess.run replacement recording the command and list file."""
    import subprocess

    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001, ANN202
        captured["cmd"] = list(cmd)
        list_arg = cmd[cmd.index("-input_file_list") + 1]
        captured["list_path"] = Path(list_arg)
        captured["list_lines"] = captured["list_path"].read_text(encoding="utf-8").splitlines()
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="boom")

    return fake_run


def test_build_vrt_passes_tiles_via_input_file_list(
    dtm_tiles: list[Path], dtm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tiles are passed via -input_file_list, not as command-line arguments:
    Windows caps the CreateProcess command line at 32767 chars, which
    thousands of tile paths exceed (WinError 206)."""
    captured: dict = {}
    monkeypatch.setattr(vrt.subprocess, "run", _fake_run_factory(captured))
    out = build_vrt(dtm_tiles, dtm_dir / "dtm.vrt")
    cmd = captured["cmd"]
    assert cmd[1:3] == ["-overwrite", "-input_file_list"]
    assert cmd[-1] == str(out)
    # No tile path on the command line; all paths (absolute) in the list file.
    assert not any(str(p) in cmd for p in dtm_tiles)
    assert captured["list_lines"] == [str(Path(p).absolute()) for p in dtm_tiles]
    # The temporary list file is removed after the call.
    assert not captured["list_path"].exists()


def test_build_vrt_many_tiles_stays_under_command_line_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for WinError 206 with 2000-3000 tiles: even with long paths
    the spawned command line stays far below the 32767-char CreateProcess
    limit, and every tile is in the input list file."""
    captured: dict = {}
    monkeypatch.setattr(vrt.subprocess, "run", _fake_run_factory(captured))
    tiles = [tmp_path / f"very_long_tile_name_{i:05d}_DTM.tif" for i in range(3000)]
    build_vrt(tiles, tmp_path / "out.vrt")
    assert len(" ".join(captured["cmd"])) < 32767
    assert len(captured["list_lines"]) == 3000
    assert not captured["list_path"].exists()


def test_build_vrt_failure_raises_and_cleans_up(
    dtm_tiles: list[Path], dtm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}
    monkeypatch.setattr(vrt.subprocess, "run", _fake_run_factory(captured, returncode=1))
    with pytest.raises(RuntimeError, match="boom"):
        build_vrt(dtm_tiles, dtm_dir / "x.vrt")
    assert not captured["list_path"].exists()
