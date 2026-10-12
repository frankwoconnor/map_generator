# Regions: working around large PBF files

A country-sized OpenStreetMap extract (e.g. `ireland-and-northern-ireland-latest.osm.pbf`,
~354 MB, 47 M nodes) is slow to read for every map. The **region library** clips it once
into small, named area files that generate in seconds and are easy to reuse.

Clipping is done by [`osmium-tool`](https://osmcode.org/osmium-tool/), which streams the
source file. Check it is installed with:

```bash
make doctor
```

## Quick start

```bash
# One-off: clip Cork out of the big file and register it
mapgen regions bootstrap \
    --from ../data/ireland-and-northern-ireland-latest.osm.pbf \
    --city cork

# See what you have
mapgen regions list

# Generate from the region (centre/distance are derived from its bbox)
mapgen generate --region cork --out ../output --prefix cork
```

Regions live in `../data/regions/` (outside the repo, so they can be synced to the Pi),
indexed by `../data/regions/regions.json`:

```
../data/regions/
├── regions.json      # catalog: name, bbox, size, counts, source, created
├── cork.osm.pbf      # 12.5 MB
├── dublin.osm.pbf
└── ...
```

## Commands

```bash
# Clip an arbitrary area (centre + half-width in metres)
mapgen regions extract --name "Killarney" \
    --center 52.0599 -9.5044 --dist 6000 \
    --from ../data/ireland-and-northern-ireland-latest.osm.pbf

# Clip every city in config/regions/ireland_cities.json (~15 s each)
mapgen regions bootstrap --from ../data/ireland-and-northern-ireland-latest.osm.pbf --all

# Inspect
mapgen regions info cork
mapgen regions info cork --json
mapgen info ../data/regions/cork.osm.pbf      # same info for any PBF, no catalog

# Remove (add --delete-file to also delete the PBF)
mapgen regions remove cork --delete-file
```

The city table is editable at `config/regions/ireland_cities.json` — each entry is a
name, `center: [lat, lon]` and an optional `radius_m` (default 6000, giving a 12×12 km box).

## Automatic clipping (no region needed)

You do not have to pre-build a region. If you point the generator at a large PBF with a
centre and distance, it clips the needed area automatically and caches the result under
`../cache/clips/`:

```bash
python main.py --pbf ../data/ireland-and-northern-ireland-latest.osm.pbf \
    --center 51.8986 -8.4756 --dist 1000 --prefix cork_auto
```

The first run clips (~15 s for a 1 km area, producing a ~2 MB file); later runs over the
same area reuse the cached clip. Behaviour is controlled by `processing` in `style.json`:

```json
"processing": {
  "auto_clip": true,
  "auto_clip_threshold_mb": 20.0,
  "auto_clip_margin": 0.2
}
```

- `auto_clip` — enable automatic clipping for files above the threshold.
- `auto_clip_threshold_mb` — only clip files larger than this.
- `auto_clip_margin` — grow the requested bbox by this fraction when clipping, so edges
  and complete features are not lost.

If `osmium` is not installed the generator falls back to reading the full PBF (with a
warning); install osmium to make big files practical.

## Tips

- Prefer regions for repeated work: they are small, fast, and portable.
- `--dist` is a **half-width**, so `--dist 1000` gives a 2×2 km map.
- For coastlines and water, clip a little larger than the final map and crop at render
  time, so features crossing the border stay whole.
