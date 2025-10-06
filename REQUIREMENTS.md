# Requirements Specification

## Project Structure
```
Map_gen_v2/
├── app/                    # Flask application code
├── config/                 # Configuration files
│   ├── layers/            # Layer-specific configurations
│   ├── palettes/          # Color palettes
│   └── schemas/           # JSON schemas for configuration validation
├── docs/                  # Documentation
├── map_core/              # Core map generation logic
├── output/                # Generated map outputs
├── static/                # Static assets (CSS, JS, images)
├── templates/             # HTML templates
├── tests/                 # Test suite
│   ├── integration/      # Integration tests
│   └── unit/             # Unit tests
└── tools/                 # Utility scripts
```

## Functional Requirements
- FR1: Generate per-layer SVGs (streets, buildings, water, green) and a combined SVG for a given location
- FR2: Support building color modes: manual, auto-size, auto-distance, manual floorsize categories
- FR3: Allow configuration of per-layer styling (facecolor, edgecolor, linewidth, alpha, z-order)
- FR4: Apply circular distance clipping uniformly across layers
- FR5: Produce optimized SVGs while preserving originals
- FR6: Export high-quality PNGs from either the original or optimized SVG
- FR7: Allow external CSS extraction and ensure editor-friendly outputs
- FR8: Provide a modern web UI for configuration and generation
- FR9: Support for custom layer tags and OSM query customization

## Non-Functional Requirements
- NFR1: Outputs remain highly editable in vector editors (e.g., Inkscape)
- NFR2: Optimized SVG size should be significantly smaller than originals when CSS extraction is enabled
- NFR3: Gracefully handle missing/unsupported optimizer flags across Scour versions
- NFR4: Config-driven pipeline with sane defaults; safe fallbacks to avoid failures
- NFR5: Maintainable code structure with modular helpers and type hints
- NFR6: Comprehensive test coverage with unit and integration tests
- NFR7: Responsive web UI that works on desktop and mobile devices

## Configuration

### Main Configuration
- `config/style.json`: Main style configuration for map generation
- `config/svg_optimize.json`: Controls SVG post-processing pipeline
- `config/layers/layer_tags.json`: Defines OSM tags for different layers
- `config/palettes/`: Directory containing color palette definitions

### Configuration Management
- Managed through `ConfigManager` in `config/manager.py`
- Supports loading configurations from JSON files with schema validation
- Automatic creation of default configurations if not present

## Environment Requirements
- Python 3.12+
- Core dependencies (see `requirements.txt` for exact versions):
  - osmnx
  - matplotlib
  - flask
  - jinja2
  - pandas
  - numpy
  - scipy
  - requests
  - pillow
  - cairosvg
  - lxml
  - scour

## Development Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   python app.py
   ```

4. Access the web interface at `http://localhost:5000`

## Testing
- Run all tests:
  ```bash
  python -m pytest tests/
  ```

- Run with coverage report:
  ```bash
  python -m pytest --cov=. tests/
  ```

## Output Structure
- Generated files are saved in the parent directory's `output/` folder
- Each run creates a timestamped directory
- Optimized outputs are saved in `output/<run>/optimized/` with `_opt` suffix
- PNG exports are generated alongside their source SVGs

## Quality Goals
- Clean, maintainable code with type hints
- Comprehensive test coverage
- Clear documentation
- Consistent code style (enforced by linters)
- Responsive and accessible web interface
