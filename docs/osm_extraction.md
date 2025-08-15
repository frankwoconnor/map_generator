# Extracting Cork from the Ireland OSM PBF

This guide shows how to extract a Cork-area PBF from a larger Ireland PBF while preserving relations (important for lakes like "The Lough"). It also lists useful options.

## Prerequisites

- macOS: `brew install osmium-tool`
- Source file available at: `../osm-data/ireland.pbf`
- Cork polygon file exists at: `tools/cork.poly`

## Recommended (polygon-based) extraction

Preserves complete multipolygons/relations and sets a clean bounding box.

```bash
osmium extract --polygon tools/cork.poly \
  --set-bounds --strategy=smart \
  -o ../osm-data/ireland_cork.osm.pbf \
  ../osm-data/ireland.pbf
```

### Option reference
- `--polygon <file.poly>`: clip using a .poly file (preferred for correctness).
- `--set-bounds`: sets the bounding box on the output file to the clipping polygon/bbox.
- `--strategy=smart`: keeps necessary related objects (relations, multipolygons) intact during extraction.
- `-o <output>`: output PBF path.

## Quick alternative (bbox-based) extraction

Faster, but less precise than polygon clipping. Still uses `--strategy=smart`.

```bash
osmium extract --bbox -8.55,51.84,-8.43,51.93 \
  --set-bounds --strategy=smart \
  -o ../osm-data/cork_bbox_wide.osm.pbf \
  ../osm-data/ireland.pbf
```

### Option reference
- `--bbox <min_lon,min_lat,max_lon,max_lat>`: rectangular clip.

## Optional: prefilter to water features (reduce size)

You can optionally prefilter water features, then polygon-clip.

```bash
osmium tags-filter ../osm-data/ireland.pbf -o ../osm-data/ireland_water.osm.pbf \
  nwr/natural=water nwr/water=lake nwr/landuse=water nwr/landuse=reservoir \
  nwr/waterway=riverbank nwr/leisure=marina

osmium extract --polygon tools/cork.poly \
  --set-bounds --strategy=smart \
  -o ../osm-data/cork_water.osm.pbf \
  ../osm-data/ireland_water.osm.pbf
```

## Verify "The Lough" exists in the extract

Use the script added to this repo to verify by name:

```bash
python3 scripts/find_lough_in_pbf.py ../osm-data/ireland_cork.osm.pbf \
  --name "The Lough" --contains \
  --center 51.894333 -8.480534 --dist 3000 \
  --save output/the_lough.geojson
```

If `pyrosm` is not installed, use the one-step shell script (falls back to `osmium + jq`):

```bash
bash scripts/extract_and_verify_lough.sh
```
