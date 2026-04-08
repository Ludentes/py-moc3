"""Round-trip tests for moc3 reader/writer."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from moc3 import Moc3, MocVersion, CanvasInfo


def test_empty_moc3_roundtrip():
    """Create a minimal moc3 in memory and verify round-trip."""
    moc = Moc3()
    moc.header.version = MocVersion.V3_00
    moc.canvas.pixels_per_unit = 100.0
    moc.canvas.canvas_width = 1024.0
    moc.canvas.canvas_height = 1024.0

    data = moc.to_bytes()
    moc2 = Moc3.from_bytes(data)

    assert moc2.header.version == MocVersion.V3_00
    assert moc2.canvas.pixels_per_unit == 100.0
    assert moc2.canvas.canvas_width == 1024.0

    # Second round-trip should be identical
    data2 = moc2.to_bytes()
    assert data == data2


def test_magic_validation():
    """Reject files without MOC3 magic."""
    with pytest.raises(ValueError, match="Not a MOC3 file"):
        Moc3.from_bytes(b"NOT3" + b"\x00" * 1000)


def test_modify_and_roundtrip():
    """Modify a section and verify it persists through read/write."""
    moc = Moc3()
    moc.header.version = MocVersion.V3_00
    moc.counts[0] = 2  # 2 parts
    moc["part.ids"] = ["PartA", "PartB"]
    moc["part.runtime_space"] = []
    moc["part.keyform_binding_band_indices"] = [0, 1]
    moc["part.keyform_begin_indices"] = [0, 0]
    moc["part.keyform_counts"] = [1, 1]
    moc["part.visibles"] = [True, True]
    moc["part.enables"] = [True, False]
    moc["part.parent_part_indices"] = [-1, 0]

    data = moc.to_bytes()
    moc2 = Moc3.from_bytes(data)

    assert moc2["part.ids"] == ["PartA", "PartB"]
    assert moc2["part.enables"] == [True, False]
    assert moc2["part.parent_part_indices"] == [-1, 0]


# -- File-based tests (skipped if no moc3 file available) --

MOC3_PATHS = [
    Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/Live2DModels/hiyori_vts/hiyori.moc3",
]


def _find_moc3() -> Path | None:
    for p in MOC3_PATHS:
        if p.exists():
            return p
    return None


@pytest.fixture
def moc3_file() -> Path:
    p = _find_moc3()
    if p is None:
        pytest.skip("No .moc3 test file found")
    return p


def test_file_roundtrip(moc3_file: Path):
    """Read a real moc3 file, write it back, verify byte-identical."""
    original = moc3_file.read_bytes()
    moc = Moc3.from_bytes(original)
    written = moc.to_bytes()
    assert len(written) == len(original)
    assert hashlib.sha256(written).hexdigest() == hashlib.sha256(original).hexdigest()


def test_file_info(moc3_file: Path):
    """Verify basic parsing of a real moc3 file."""
    moc = Moc3.from_file(moc3_file)
    assert moc.header.version >= MocVersion.V3_00
    assert len(moc.part_ids) > 0
    assert len(moc.art_mesh_ids) > 0
    assert len(moc.parameter_ids) > 0
    assert moc.canvas.canvas_width > 0
    assert moc.canvas.canvas_height > 0


def test_file_modify_roundtrip(moc3_file: Path):
    """Modify a value, write, read back, verify the change persists."""
    original = moc3_file.read_bytes()
    moc = Moc3.from_bytes(original)

    # Change first art mesh texture index
    orig_tex = moc["art_mesh.texture_indices"][0]
    moc["art_mesh.texture_indices"][0] = 99

    modified = moc.to_bytes()
    assert modified != original

    moc2 = Moc3.from_bytes(modified)
    assert moc2["art_mesh.texture_indices"][0] == 99

    # Change it back and verify we get the original
    moc2["art_mesh.texture_indices"][0] = orig_tex
    restored = moc2.to_bytes()
    assert restored == original
