# py-moc3

Python reader/writer for Live2D Cubism `.moc3` binary files. Zero dependencies — just Python stdlib.

## Install

```bash
pip install py-moc3
```

## Quick Start

```python
from moc3 import Moc3

# Read
moc = Moc3.from_file("model.moc3")
print(moc.summary())
print(moc.parameter_ids)
print(moc.art_mesh_ids)

# Modify
moc["art_mesh.texture_indices"][0] = 1
moc["parameter.max_values"][0] = 60.0

# Write
moc.to_file("model_modified.moc3")
```

## CLI

```bash
# Show model info
moc3 info model.moc3

# List parameters with ranges
moc3 params model.moc3
moc3 params model.moc3 --json

# List art meshes
moc3 meshes model.moc3

# Verify read-write round-trip (byte-identical check)
moc3 verify model.moc3
```

## Section Access

All data is stored in named sections using struct-of-arrays layout. Access any section by name:

```python
moc = Moc3.from_file("model.moc3")

# IDs
moc["part.ids"]           # list[str]
moc["deformer.ids"]       # list[str]
moc["art_mesh.ids"]       # list[str]
moc["parameter.ids"]      # list[str]

# Art mesh properties
moc["art_mesh.texture_indices"]   # list[int] — which texture sheet
moc["art_mesh.vertex_counts"]    # list[int]
moc["art_mesh.visibles"]         # list[bool]

# Parameter ranges
moc["parameter.min_values"]      # list[float]
moc["parameter.max_values"]      # list[float]
moc["parameter.default_values"]  # list[float]

# UV coordinates (flat x,y pairs)
moc["uv.xys"]                    # list[float]

# Triangle indices
moc["position_index.indices"]    # list[int]

# Canvas info
moc.canvas.canvas_width          # float
moc.canvas.canvas_height         # float
moc.canvas.pixels_per_unit       # float
```

See `SECTION_LAYOUT` in the source for the complete list of ~100 available sections.

## Format Support

| Version | Read | Write | Notes |
|---------|------|-------|-------|
| V3.00 (1) | Yes | Yes | Base format |
| V3.03 (2) | Yes | Yes | + quad transforms |
| V4.00 (3) | Yes | Yes | |
| V4.02 (4) | Yes | Yes | |
| V5.00 (5) | Yes | Yes | |

Round-trip tested: read → write produces byte-identical output.

## How It Works

The `.moc3` format is a flat binary with:

1. **Header** (64 bytes): magic `"MOC3"`, format version, endian flag
2. **Section Offset Table** (640 bytes): 160 × uint32 offsets pointing into the body
3. **Body** (from byte 1984): ~100 typed arrays, each 64-byte aligned

Each array stores one property for all items of a type (struct-of-arrays). For example, `art_mesh.texture_indices` stores the texture index for every art mesh, contiguously.

Ported from [moc3-reader-re](https://github.com/YusaeMiu/moc3-reader-re) (Java decompilation of the Cubism SDK exporter).

## License

MIT
