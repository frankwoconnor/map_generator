# Requirements Specification

## Functional Requirements
- FR1: Generate per-layer SVGs (streets, buildings, water, green) and a combined SVG for a given location.
- FR2: Support building color modes: manual, auto-size, auto-distance, manual floorsize categories.
- FR3: Allow configuration of per-layer styling (facecolor, edgecolor, linewidth, alpha, z-order) and building-specific facecolor logic.
- FR4: Apply circular distance clipping uniformly across layers.
- FR5: Produce optimized SVGs while preserving originals.
- FR6: Export high-quality PNGs from either the original or optimized SVG.
- FR7: Allow external CSS extraction and ensure editor-friendly outputs.
- FR8: Provide a web UI for configuration and generation.

## Non-Functional Requirements
- NFR1: Outputs remain highly editable in vector editors (e.g., Inkscape).
- NFR2: Optimized SVG size should be significantly smaller than originals when CSS extraction is enabled.
- NFR3: Gracefully handle missing/unsupported optimizer flags across Scour versions.
- NFR4: Config-driven pipeline with sane defaults; safe fallbacks to avoid failures.
- NFR5: Maintainable code structure with modular helpers and type hints.
- NFR6: Unit tests for utilities and palette/category logic; smoke tests for pipeline.

## Configuration Requirements
- JSON schemas define valid configuration for `style.json`. Unknown fields are ignored or normalized.
- `svg_optimize.json` controls post-processing with documented options for optimizer, CSS, PNG export, and experimental steps.

## Environment Requirements
- Python 3.10+
- Pinned Python dependencies in `requirements.txt`:
  - osmnx>=2.0,<3.0
  - matplotlib==3.10.5
  - scour==0.38.2
  - cairosvg>=2.7.0,<3.0
  - lxml>=5.2,<6
  - flask>=3.0,<4
- Optional system deps: libcairo for CairoSVG; see platform docs.

## Output Requirements
- Originals saved under `output/<run>/`.
- Optimized copies under `output/<run>/optimized/` with `_opt` suffix.
- PNGs saved adjacent to the selected source with `_hq` suffix.

## Quality Goals
- Readable, minimal SVG output with preserved semantics.
- Deterministic pipeline behavior across runs with same inputs.
- Clear error messages and logs for diagnosis.
