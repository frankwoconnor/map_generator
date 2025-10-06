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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import json
import os
import hashlib

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
    cfg_path = path or DEFAULT_OPTIMIZE_FILE
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def optimize_svg_file(svg_path: str, out_path: Optional[str], config: Dict[str, Any]) -> OptimizeResult:
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
    if target_svg is None:
        write_separate = bool(config.get("write_separate", False))
        if write_separate:
            base_dir = os.path.dirname(svg_path)
            base_name = os.path.basename(svg_path)
            name, ext = os.path.splitext(base_name)
            suffix = str(config.get("optimized_suffix", "_opt"))
            opt_dir_cfg = config.get("optimized_dir")
            if isinstance(opt_dir_cfg, str) and opt_dir_cfg:
                # If relative path, place under original directory
                opt_dir = opt_dir_cfg if os.path.isabs(opt_dir_cfg) else os.path.join(base_dir, opt_dir_cfg)
            else:
                opt_dir = base_dir
            os.makedirs(opt_dir, exist_ok=True)
            target_svg = os.path.join(opt_dir, f"{name}{suffix}{ext}")
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

    return OptimizeResult(optimized_svg=optimized, css_written=css_written, png_written=png_written)


def _optimize_with_scour(svg_text: str, opts: Dict[str, Any]) -> str:
    if scour is None:
        # Scour not installed; return original
        return svg_text

    # Map config options to scour CLI args (conservative set for wide compatibility)
    # We'll avoid flags known to vary between versions and add a fallback path.
    args: List[str] = []

    def flag(opt_key: str, true_flag: str, false_flag: Optional[str] = None, default: Optional[bool] = None):
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


def _extract_css(svg_text: str, svg_path: str, cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Extract inline style attributes into an external CSS file and apply classes.

    Only a conservative subset of attributes are extracted to keep editability.
    """
    if etree is None:
        return svg_text, None

    attrs: List[str] = cfg.get("attributes", [
        "fill", "stroke", "stroke-width", "opacity",
        "fill-opacity", "stroke-opacity", "vector-effect",
        "stroke-linejoin", "stroke-linecap", "stroke-dasharray",
    ])
    merge_rules = bool(cfg.get("merge_rules", True))
    css_filename = cfg.get("file", "styles.css")

    try:
        parser = etree.XMLParser(remove_blank_text=False, ns_clean=True, recover=True)
        root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)
        nsmap = root.nsmap.copy() if hasattr(root, 'nsmap') else {}

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
            tag = etree.QName(el).localname if hasattr(etree, 'QName') else el.tag
            if tag in ("style",):
                continue
            collect_style(el)

        # Create <link> to external CSS
        css_path = os.path.join(os.path.dirname(svg_path), css_filename)
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


def _export_png(svg_text: str, base_svg_path: str, cfg: Dict[str, Any]) -> Optional[str]:
    if cairosvg is None:
        return None
    # Determine output path
    suffix = cfg.get("filename_suffix", "_hq")
    out_path = os.path.splitext(base_svg_path)[0] + f"{suffix}.png"

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
        nsmap = root.nsmap.copy() if hasattr(root, 'nsmap') else {}
        svg_ns = nsmap.get(None, "http://www.w3.org/2000/svg")
        ns = {"svg": svg_ns}

        # Remove clipPath defs
        for defs in root.findall(".//svg:defs", namespaces=ns):
            for cp in list(defs.findall("svg:clipPath", namespaces=ns)):
                defs.remove(cp)

        # Strip clip-path attrs on all elements
        for el in root.iter():
            if 'clip-path' in el.attrib:
                try:
                    del el.attrib['clip-path']
                except Exception:
                    pass

        return etree.tostring(root, encoding="utf-8").decode("utf-8")
    except Exception:
        return svg_text
