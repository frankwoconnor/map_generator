# `mapgen` command line

`mapgen` is the fast path to making map art without opening the web UI or hand-editing
`style.json`. It is installed as a console script (`pip install -e .`); you can also run
it as `python -m mapgen`.

```
mapgen info      <pbf>                              inspect a PBF
mapgen regions   list | info | extract | bootstrap | remove
mapgen generate  --region <slug> | --pbf <file>     generate artwork
```

Global help: `mapgen --help`, and per-command: `mapgen generate --help`.

## Inspecting data

```bash
mapgen info ../data/regions/cork.osm.pbf
# ../data/regions/cork.osm.pbf
#   size:      12.5 MB (13,054,954 bytes)
#   bbox:      (-8.6037107, 51.8195486, -8.3474892, 51.9776514)
#   nodes:     1,597,305
#   ways:      280,317
#   relations: 1,381
#   source:    osmium/1.19.1
#   updated:   2025-08-08T19:54:57Z

mapgen info ../data/regions/cork.osm.pbf --json   # machine-readable
```

## Generating

```bash
# From a catalogued region (centre and distance come from the region bbox)
mapgen generate --region cork --out ../output --prefix cork

# From an explicit PBF and area
mapgen generate --pbf ../data/regions/cork.osm.pbf \
    --center 51.8986 -8.4756 --dist 1500 \
    --out ../output --prefix cork_centre

# Only some layers
mapgen generate --region cork --layers streets,water
```

## Style overrides with `--set`

Override any field of `style.json` for a single run. The path must already exist
(typos fail loudly with the list of valid keys at that level). Values are coerced to
bool/int/float/list where obvious.

```bash
mapgen generate --region cork \
    --set layers.buildings.facecolor=#ff0066 \
    --set layers.streets.linewidth=2 \
    --set output.background_color=#101018 \
    --set layers.water.enabled=false
```

`--set` may be repeated; later values win. The run's effective config is saved to the
output directory as usual.

## Regions

See [regions.md](regions.md) for the full walkthrough:

```bash
mapgen regions bootstrap --from ../data/ireland-and-northern-ireland-latest.osm.pbf --city cork,dublin
mapgen regions list
mapgen regions info cork --json
```

## Notes

- `generate` currently drives the same generator as `main.py`; the flags mirror the
  `--pbf/--center/--dist/--out/--prefix/--layers` options there.
- Everything printed by the generator goes to stdout/stderr, so it composes with other
  tools: `mapgen regions list --json | jq '.[].slug'`.
