"""SVG post-processing utilities.

Features:
- Config-driven optimization using Python tools (Scour preferred).
- Optional CSS extraction to external file for style attributes.
- Optional high-quality PNG export from the optimized SVG.

Design:
- Load optional config from svg_optimize.json at project root (or provided path).
- Gracefully degrade if optional dependencies are missing (scour, cairosvg, lxml).
- Keep output editable for vector editors (e.g., Inkscape) by not over-aggressively simplifying.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps loaded lazily
try:
    from scour import scour  # type: ignore
except Exception:  # pragma: no cover
    scour = None  # type: ignore

try:
    import cairosvg  # type: ignore
except Exception:  # pragma: no cover
    cairosvg = None  # type: ignore

try:
    from lxml import etree  # type: ignore
except Exception:  # pragma: no cover
    etree = None  # type: ignore

try:
    # lightweight logging helper from core.util if available
    from map_core.core.util import log_progress  # type: ignore
except Exception:  # pragma: no cover

    def log_progress(msg: str) -> None:
        try:
            print(msg)
        except Exception:
            pass


DEFAULT_OPTIMIZE_FILE = "config/svg_optimize.json"


@dataclass
class OptimizeResult:
    optimized_svg: str
    css_written: Optional[str] = None
    png_written: Optional[str] = None


def load_optimize_config(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    cfg_path = Path(path or DEFAULT_OPTIMIZE_FILE)
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def optimize_svg_file(
    svg_path: str, out_path: Optional[str], config: Dict[str, Any]
) -> OptimizeResult:
    """Optimize a single SVG path according to config and write outputs.

    Args:
        svg_path: Path to original SVG file.
        out_path: Path to write the optimized SVG. If None, overwrite original.
        config: Config dictionary loaded from svg_optimize.json.
    """
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_text = f.read()

    enabled = bool(config.get("enabled", True))
    if not enabled:
        # no-op
        if out_path and out_path != svg_path:
            with open(out_path, "w", encoding="utf-8") as wf:
                wf.write(svg_text)
            return OptimizeResult(optimized_svg=svg_text)
        return OptimizeResult(optimized_svg=svg_text)

    # Determine target output path (where optimized SVG will be written)
    target_svg = out_path
    svg_path_obj = Path(svg_path)
    if target_svg is None:
        write_separate = bool(config.get("write_separate", False))
        if write_separate:
            name = svg_path_obj.stem
            ext = svg_path_obj.suffix
            suffix = str(config.get("optimized_suffix", "_opt"))
            opt_dir_cfg = config.get("optimized_dir")
            if isinstance(opt_dir_cfg, str) and opt_dir_cfg:
                opt_dir_path = Path(opt_dir_cfg)
                # If relative path, place under original directory
                opt_dir = (
                    opt_dir_path
                    if opt_dir_path.is_absolute()
                    else svg_path_obj.parent / opt_dir_path
                )
            else:
                opt_dir = svg_path_obj.parent
            opt_dir.mkdir(parents=True, exist_ok=True)
            target_svg = str(opt_dir / f"{name}{suffix}{ext}")
        else:
            target_svg = svg_path

    # Step 1: Scour optimization (if selected)
    optimizer = config.get("optimizer", "scour")
    optimized = svg_text

    if optimizer == "scour":
        optimized = _optimize_with_scour(svg_text, config.get("scour", {}) or {})
    elif optimizer == "none":
        optimized = svg_text
    else:
        # Unknown optimizer; pass-through
        optimized = svg_text

    # Optional experimental: remove clipPaths/clip-path attributes
    exp_cfg = config.get("experimental", {}) or {}
    if exp_cfg.get("remove_clip_paths", False):
        optimized = _remove_clip_paths(optimized)

    css_written = None
    css_cfg = config.get("css", {}) or {}
    if css_cfg.get("extract", False):
        optimized, css_written = _extract_css(optimized, target_svg, css_cfg)
        if css_written:
            log_progress(f"CSS extracted to: {css_written}")

    # Write optimized svg
    with open(target_svg, "w", encoding="utf-8") as wf:
        wf.write(optimized)
    log_progress(f"Optimized SVG written: {target_svg}")

    png_written = None
    png_cfg = config.get("png_export", {}) or {}
    if png_cfg.get("enabled", False):
        source = str(png_cfg.get("source", "optimized")).lower()
        if source == "original":
            base_path = svg_path
            src_text = svg_text
        else:
            base_path = target_svg
            src_text = optimized
        png_written = _export_png(src_text, base_path, png_cfg)
        if png_written:
            log_progress(f"PNG exported: {png_written}")
        else:
            log_progress("PNG export failed (see earlier logs if any)")

    return OptimizeResult(
        optimized_svg=optimized, css_written=css_written, png_written=png_written
    )


def _optimize_with_scour(svg_text: str, opts: Dict[str, Any]) -> str:
    if scour is None:
        # Scour not installed; return original
        return svg_text

    # Map config options to scour CLI args (conservative set for wide compatibility)
    # We'll avoid flags known to vary between versions and add a fallback path.
    args: List[str] = []

    def flag(
        opt_key: str,
        true_flag: str,
        false_flag: Optional[str] = None,
        default: Optional[bool] = None,
    ):
        val = opts.get(opt_key, default)
        if val is None:
            return
        if bool(val):
            args.append(true_flag)
        elif false_flag:
            args.append(false_flag)

    def kv(opt_key: str, flag_name: str, conv=lambda v: str(v)):
        if opt_key in opts and opts.get(opt_key) is not None:
            args.extend([flag_name, conv(opts.get(opt_key))])

    # Common scour options    # Booleans (avoid --strip-comments and other version-fragile flags)
    flag("remove_metadata", "--remove-metadata")
    flag("enable_viewboxing", "--enable-viewboxing")
    flag("keep_editor_data", "--keep-editor-data")
    flag("renderer_workaround", "--renderer-workaround")
    flag("remove_descriptive_elements", "--remove-descriptive-elements")
    flag("shorten_ids", "--shorten-ids", default=False)
    kv("digits", "--set-precision", lambda v: str(int(v)))
    # pretty/indent settings
    indent_type = opts.get("indent_type", "none")
    if indent_type == "space":
        args.extend(["--indent=space", "--nindent", str(int(opts.get("nindent", 2)))])
    elif indent_type == "tab":
        args.append("--indent=tab")
    else:
        args.append("--indent=none")

    def run_with(arg_list: List[str]) -> Optional[str]:
        try:
            parsed = scour.parse_args(arg_list)
            options = scour.sanitizeOptions(parsed)
            return scour.scourString(svg_text, options=options)
        except Exception:
            return None

    # First attempt with full args
    out = run_with(args)
    if out is not None:
        return out

    # Fallback to minimal, broadly supported flags
    minimal: List[str] = []
    if bool(opts.get("remove_metadata", True)):
        minimal.append("--remove-metadata")
    if bool(opts.get("enable_viewboxing", True)):
        minimal.append("--enable-viewboxing")
    # precision and indent
    if "digits" in opts:
        try:
            minimal.extend(["--set-precision", str(int(opts.get("digits", 3)))])
        except Exception:
            pass
    minimal.append("--indent=none")

    out2 = run_with(minimal)
    if out2 is not None:
        return out2

    # Last resort: return original
    return svg_text


def _extract_css(
    svg_text: str, svg_path: str, cfg: Dict[str, Any]
) -> Tuple[str, Optional[str]]:
    """Extract inline style attributes into an external CSS file and apply classes.

    Only a conservative subset of attributes are extracted to keep editability.
    """
    if etree is None:
        return svg_text, None

    attrs: List[str] = cfg.get(
        "attributes",
        [
            "fill",
            "stroke",
            "stroke-width",
            "opacity",
            "fill-opacity",
            "stroke-opacity",
            "vector-effect",
            "stroke-linejoin",
            "stroke-linecap",
            "stroke-dasharray",
        ],
    )
    merge_rules = bool(cfg.get("merge_rules", True))
    css_filename = cfg.get("file", "styles.css")

    try:
        parser = etree.XMLParser(remove_blank_text=False, ns_clean=True, recover=True)
        root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)
        nsmap = root.nsmap.copy() if hasattr(root, "nsmap") else {}

        # Build style map
        style_to_class: Dict[str, str] = {}
        rules: Dict[str, Dict[str, str]] = {}

        def style_key(d: Dict[str, str]) -> str:
            # stable key for dict of attrs
            items = sorted((k, v) for k, v in d.items())
            return hashlib.sha1(json.dumps(items).encode("utf-8")).hexdigest()[:8]

        def collect_style(el) -> Optional[str]:
            style_map: Dict[str, str] = {}
            for a in attrs:
                val = el.get(a)
                if val is not None:
                    style_map[a] = val
            if not style_map:
                return None
            if merge_rules:
                k = style_key(style_map)
                cls = style_to_class.get(k)
                if cls is None:
                    cls = f"s-{k}"
                    style_to_class[k] = cls
                    rules[cls] = style_map
            else:
                # unique per element
                k = style_key(style_map)
                cls = f"s-{k}"
                rules[cls] = style_map
            # apply class
            prior_class = el.get("class")
            el.set("class", (prior_class + " " if prior_class else "") + cls)
            # remove attributes moved to CSS
            for a in style_map.keys():
                if el.get(a) is not None:
                    del el.attrib[a]
            return cls

        # Walk elements
        for el in root.iter():
            # skip <style> or defs that already contain CSS
            tag = etree.QName(el).localname if hasattr(etree, "QName") else el.tag
            if tag in ("style",):
                continue
            collect_style(el)

        # Create <link> to external CSS
        css_path = str(Path(svg_path).parent / css_filename)
        # Insert <link> after root tag start
        head_link = etree.Element("link")
        head_link.set("rel", "stylesheet")
        head_link.set("type", "text/css")
        head_link.set("href", css_filename)
        # For SVG, external stylesheets can be referenced using processing instructions or <style>@import</style>.
        # Here, we embed a <style>@import</style> for broad tool compatibility.
        style_import = etree.Element("style")
        style_import.text = f"@import url('{css_filename}');"
        # Insert at top
        root.insert(0, style_import)

        # Serialize updated SVG
        new_svg = etree.tostring(root, encoding="utf-8").decode("utf-8")

        # Ensure the SVG references the external CSS via <style>@import
        css_file = cfg.get("file", "styles.css")

        # Insert a <style> tag early under <svg> or inside a <defs>
        def ensure_css_import(root_el) -> None:
            # Check if already present
            for el in root_el.findall(".//{http://www.w3.org/2000/svg}style"):
                if el.text and css_file in el.text:
                    return
            # Create/insert style element
            style_el = etree.Element("style")
            style_el.set("type", "text/css")
            style_el.text = f"@import url('{css_file}');"
            # Prefer defs if available
            defs_el = root_el.find("{http://www.w3.org/2000/svg}defs")
            if defs_el is not None:
                defs_el.insert(0, style_el)
            else:
                # insert as first child
                root_el.insert(0, style_el)

        ensure_css_import(root)

        # Write CSS rules
        lines: List[str] = []
        for cls, d in rules.items():
            parts = [f"{k}: {v};" for k, v in d.items()]
            lines.append(f".{cls} {{ {' '.join(parts)} }}")
        if lines:
            with open(css_path, "w", encoding="utf-8") as cf:
                cf.write("\n".join(lines) + "\n")
            return new_svg, css_path
        else:
            return new_svg, None
    except Exception:
        # Fallback to original on any error
        return svg_text, None


def _export_png(
    svg_text: str, base_svg_path: str, cfg: Dict[str, Any]
) -> Optional[str]:
    if cairosvg is None:
        return None
    # Determine output path
    suffix = cfg.get("filename_suffix", "_hq")
    base_path = Path(base_svg_path)
    out_path = str(base_path.with_name(f"{base_path.stem}{suffix}.png"))

    # Extract params
    scale = float(cfg.get("scale", 2.0))
    dpi = int(cfg.get("dpi", 300))
    background = cfg.get("background", None)  # None to keep transparent

    try:
        # Use url= so CairoSVG can resolve relative assets (e.g., external CSS)
        cairosvg.svg2png(
            url=base_svg_path,
            write_to=out_path,
            scale=scale,
            dpi=dpi,
            background_color=background,
        )
        return out_path
    except Exception as e:
        log_progress(f"PNG export error: {type(e).__name__}: {e}")
        return None


def _remove_clip_paths(svg_text: str) -> str:
    """Remove all clipPath defs and clip-path attributes. Heuristic but helps file size and editor perf.

    This may affect visuals if real clipping is used beyond canvas. Use behind config flag only.
    """
    if etree is None:
        return svg_text
    try:
        parser = etree.XMLParser(remove_comments=True)
        root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)
        nsmap = root.nsmap.copy() if hasattr(root, "nsmap") else {}
        svg_ns = nsmap.get(None, "http://www.w3.org/2000/svg")
        ns = {"svg": svg_ns}

        # Remove clipPath defs
        for defs in root.findall(".//svg:defs", namespaces=ns):
            for cp in list(defs.findall("svg:clipPath", namespaces=ns)):
                defs.remove(cp)

        # Strip clip-path attrs on all elements
        for el in root.iter():
            if "clip-path" in el.attrib:
                try:
                    del el.attrib["clip-path"]
                except Exception:
                    pass

        return etree.tostring(root, encoding="utf-8").decode("utf-8")
    except Exception:
        return svg_text


# ============================================================================
# SVG export / optimisation / shape-simplification entry points
# (moved here from the monolithic ``main.py`` generator)
# ============================================================================

def build_svg_opt_config(style_output: Dict[str, Any]) -> Dict[str, Any]:
    """Build SVG optimization config from style.json output settings.
    
    Args:
        style_output: The output section of style.json
        
    Returns:
        Config dict compatible with svg_post.optimize_svg_file
    """
    # Scour settings
    scour_config = {
        "remove_metadata": style_output.get("svg_optimize_config", {}).get("remove_metadata", True),
        "strip_comments": style_output.get("svg_optimize_config", {}).get("strip_comments", True),
        "enable_viewboxing": style_output.get("svg_optimize_config", {}).get("enable_viewboxing", True),
        "keep_editor_data": style_output.get("svg_optimize_config", {}).get("keep_editor_data", False),
        "renderer_workaround": style_output.get("svg_optimize_config", {}).get("renderer_workaround", True),
        "remove_descriptive_elements": style_output.get("svg_optimize_config", {}).get("remove_descriptive_elements", False),
        "shorten_ids": style_output.get("svg_optimize_config", {}).get("shorten_ids", True),
        "simple_colors": style_output.get("svg_optimize_config", {}).get("simple_colors", False),
        "digits": style_output.get("svg_optimize_config", {}).get("digits", 3),
        "indent_type": style_output.get("svg_optimize_config", {}).get("indent_type", "none"),
        "nindent": style_output.get("svg_optimize_config", {}).get("nindent", 2),
    }
    
    css_config = {
        "extract": style_output.get("svg_css_extract", True),
        "file": "styles.css",
        "merge_rules": style_output.get("svg_css_merge_rules", True),
        "attributes": style_output.get("svg_css_attributes", [
            "fill", "stroke", "stroke-width", "opacity", "fill-opacity",
            "stroke-opacity", "vector-effect", "stroke-linejoin",
            "stroke-linecap", "stroke-dasharray"
        ]),
    }
    
    return {
        "enabled": style_output.get("svg_optimize_enabled", True),
        "optimizer": style_output.get("svg_optimizer", "scour"),
        "scour": scour_config,
        "css": css_config,
        "write_separate": style_output.get("svg_write_separate", True),
        "optimized_suffix": style_output.get("svg_optimized_suffix", "_opt"),
        "optimized_dir": style_output.get("svg_optimized_dir", "optimized"),
    }


def export_png_from_svg(
    svg_path: str,
    output_directory: str,
    filename_prefix: str,
    dpi: int,
    scale: float,
    suffix: str,
    background: Optional[str],
    source: str,
    svg_opt_config: Dict[str, Any],
) -> None:
    """Export PNG from SVG using cairosvg.
    
    Args:
        svg_path: Path to the source SVG file
        output_directory: Output directory
        filename_prefix: File prefix for the PNG
        dpi: Base DPI for rasterization
        scale: Scale factor (multiplies DPI)
        suffix: Filename suffix for PNG
        background: Background color (None for transparent)
        source: "original" or "optimized" - which SVG to use
        svg_opt_config: SVG optimization config to find optimized file
    """
    import cairosvg
    
    # Determine which SVG to use
    if source == "optimized" and svg_opt_config.get("write_separate"):
        opt_dir = svg_opt_config.get("optimized_dir", "optimized")
        opt_suffix = svg_opt_config.get("optimized_suffix", "_opt")
        opt_path = os.path.join(
            output_directory, opt_dir, f"{filename_prefix}_combined{opt_suffix}.svg"
        )
        if os.path.exists(opt_path):
            svg_path = opt_path
            log_progress(f"[png] Using optimized SVG: {opt_path}")
        else:
            log_progress(f"[png] Optimized SVG not found, falling back to original: {opt_path}")
    
    png_path = os.path.join(output_directory, f"{filename_prefix}_combined{suffix}.png")
    effective_dpi = dpi * scale
    
    log_progress(f"[png] Exporting PNG: {png_path} at {effective_dpi} DPI")
    
    # Prepare cairosvg arguments
    cairosvg_args = {
        "url": svg_path,
        "write_to": png_path,
        "dpi": effective_dpi,
    }
    if background:
        cairosvg_args["background_color"] = background
    
    # Handle CSS for optimized SVGs - if CSS was extracted, we need to inline it
    if svg_opt_config.get("css", {}).get("extract") and source == "optimized":
        css_file = svg_opt_config.get("css", {}).get("file", "styles.css")
        css_path = os.path.join(os.path.dirname(svg_path), css_file)
        if os.path.exists(css_path):
            with open(css_path, "r") as f:
                css_content = f.read()
            cairosvg_args["css"] = css_content
            log_progress(f"[png] Inlined CSS from {css_path}")
    
    cairosvg.svg2png(**cairosvg_args)
    log_progress(f"[png] PNG export done: {png_path}")


def simplify_svg_shapes(
    svg_path: str,
    tolerance: float = 0.1,
    preserve_topology: bool = True,
    min_segment_length: float = 0.5,
    low_poly_mode: bool = False,
    low_poly_grid: float = 1.0,
) -> str:
    """Simplify SVG paths and polygons using Douglas-Peucker algorithm.
    
    Args:
        svg_path: Path to SVG file to simplify
        tolerance: Douglas-Peucker epsilon in SVG coordinate units
        preserve_topology: Prevent self-intersections
        min_segment_length: Remove segments shorter than this
        low_poly_mode: Aggressive simplification (merge vertices, snap to grid)
        low_poly_grid: Grid size for vertex snapping in low-poly mode
        
    Returns:
        Path to simplified SVG (overwrites original)
    """
    try:
        from lxml import etree
        from svgpathtools import svg2paths, wsvg, Path, Line, CubicBezier, QuadraticBezier, Arc
        import numpy as np
    except ImportError as e:
        log_progress(f"[simplify] Required dependencies not available: {e}")
        log_progress("[simplify] Install with: pip install svgpathtools lxml")
        return svg_path
    
    log_progress(f"[simplify] Simplifying SVG: {svg_path}")
    
    # Parse SVG
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(svg_path, parser)
    root = tree.getroot()
    
    # Register namespaces
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    for prefix, uri in root.nsmap.items():
        if prefix:
            ns[prefix] = uri
    
    # Find all path elements
    paths = root.findall('.//svg:path', namespaces=ns)
    polygons = root.findall('.//svg:polygon', namespaces=ns)
    polylines = root.findall('.//svg:polyline', namespaces=ns)
    
    total_elements = len(paths) + len(polygons) + len(polylines)
    if total_elements == 0:
        log_progress("[simplify] No path/polygon/polyline elements found")
        return svg_path
    
    log_progress(f"[simplify] Found {len(paths)} paths, {len(polygons)} polygons, {len(polylines)} polylines")
    
    def simplify_points(points: list, tol: float) -> list:
        """Douglas-Peucker simplification for a list of (x,y) points."""
        if len(points) <= 2:
            return points
        
        # Convert to numpy for efficiency
        pts = np.array(points)
        
        def dp_simplify(pts, epsilon):
            """Recursive Douglas-Peucker."""
            if len(pts) <= 2:
                return pts
            
            # Find point with max distance from line segment
            start, end = pts[0], pts[-1]
            if np.allclose(start, end):
                # Closed polygon - treat specially
                line_vec = end - start
                line_len = np.linalg.norm(line_vec)
                if line_len == 0:
                    return [start]
            
            # Distance from line
            if len(pts) == 2:
                return pts
                
            # Vector from start to end
            line_vec = end - start
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                return [start]
            
            line_unit = line_vec / line_len
            
            # Project all points onto line
            proj = np.dot(pts - start, line_unit)
            # Perpendicular distance
            perp = pts - start - np.outer(proj, line_unit)
            dists = np.linalg.norm(perp, axis=1)
            
            # Ignore first and last
            dists[0] = -1
            dists[-1] = -1
            
            max_idx = np.argmax(dists)
            max_dist = dists[max_idx]
            
            if max_dist > epsilon:
                # Recursively simplify both parts
                left = dp_simplify(pts[:max_idx+1], epsilon)
                right = dp_simplify(pts[max_idx:], epsilon)
                return np.vstack([left[:-1], right])
            else:
                return np.array([start, end])
        
        simplified = dp_simplify(pts, tol)
        return [(float(p[0]), float(p[1])) for p in simplified]
    
    def remove_short_segments(points: list, min_len: float) -> list:
        """Remove consecutive points that are too close."""
        if len(points) <= 2 or min_len <= 0:
            return points
        result = [points[0]]
        for p in points[1:]:
            dx = p[0] - result[-1][0]
            dy = p[1] - result[-1][1]
            if dx*dx + dy*dy >= min_len*min_len:
                result.append(p)
        return result
    
    def snap_to_grid(points: list, grid: float) -> list:
        """Snap points to grid for low-poly look."""
        if grid <= 0:
            return points
        return [(round(p[0]/grid)*grid, round(p[1]/grid)*grid) for p in points]
    
    def process_path_d(d: str) -> str:
        """Process an SVG path 'd' attribute."""
        if not d:
            return d
        
        try:
            # Parse path using svgpathtools
            path = parse_path(d)
            
            # Convert to points
            points = []
            for segment in path:
                if isinstance(segment, (Line, CubicBezier, QuadraticBezier, Arc)):
                    # Sample the curve
                    num_samples = max(2, int(segment.length() / max(tolerance, 0.01)) + 1)
                    for i in range(num_samples):
                        t = i / (num_samples - 1)
                        pt = segment.point(t)
                        points.append((pt.real, pt.imag))
                else:
                    pt = segment.point(0)
                    points.append((pt.real, pt.imag))
            
            if not points:
                return d
            
            # Simplify
            points = simplify_points(points, tolerance)
            points = remove_short_segments(points, min_segment_length)
            
            if low_poly_mode and low_poly_grid > 0:
                points = snap_to_grid(points, low_poly_grid)
            
            # Reconstruct path as polyline (simplified)
            if len(points) < 2:
                return d
            
            new_d = f"M {points[0][0]:.3f} {points[0][1]:.3f}"
            for p in points[1:]:
                new_d += f" L {p[0]:.3f} {p[1]:.3f}"
            
            return new_d
            
        except Exception as e:
            log_progress(f"[simplify] Path parse failed, keeping original: {e}")
            return d
    
    def process_polygon_points(points_str: str) -> str:
        """Process polygon/polyline points attribute."""
        if not points_str:
            return points_str
        
        try:
            # Parse "x1,y1 x2,y2 x3,y3..."
            coords = points_str.strip().split()
            points = []
            for c in coords:
                if ',' in c:
                    x, y = c.split(',')
                    points.append((float(x), float(y)))
            
            if len(points) < 3:
                return points_str
            
            # For polygons, close the loop
            is_polygon = points[0] != points[-1] if len(points) > 1 else False
            
            # Simplify
            points = simplify_points(points, tolerance)
            points = remove_short_segments(points, min_segment_length)
            
            if low_poly_mode and low_poly_grid > 0:
                points = snap_to_grid(points, low_poly_grid)
            
            # Ensure polygon is closed
            if is_polygon and points[0] != points[-1]:
                points.append(points[0])
            
            return ' '.join(f"{p[0]:.3f},{p[1]:.3f}" for p in points)
            
        except Exception as e:
            log_progress(f"[simplify] Polygon parse failed: {e}")
            return points_str
    
    # Helper: parse_path from svgpathtools
    def parse_path(d):
        from svgpathtools import parse_path as sp_parse_path
        return sp_parse_path(d)
    
    # Process all paths
    modified = 0
    for path in paths:
        d = path.get('d')
        if d:
            new_d = process_path_d(d)
            if new_d != d:
                path.set('d', new_d)
                modified += 1
    
    for poly in polygons:
        points = poly.get('points')
        if points:
            new_points = process_polygon_points(points)
            if new_points != points:
                poly.set('points', new_points)
                modified += 1
    
    for poly in polylines:
        points = poly.get('points')
        if points:
            new_points = process_polygon_points(points)
            if new_points != points:
                poly.set('points', new_points)
                modified += 1
    
    log_progress(f"[simplify] Modified {modified} elements")
    
    # Write back
    tree.write(svg_path, encoding='utf-8', xml_declaration=True, pretty_print=True)
    return svg_path
