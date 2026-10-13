"""Playwright test: the new Min Water Width (sliver) filter works from the UI.

Loads the studio, enables only water+streets (fast preview), then POSTs the real
form to /api/preview with water_min_width 0 vs a large value and counts the
filled water-area polygons (#1f24bd) in the returned SVG. Asserts the filter
removes thin water slivers end-to-end (form field -> style -> fetch -> plot).
"""
import os
import re
import json
import sys
from playwright.sync_api import sync_playwright

BASE = os.environ.get("PW_BASE", "http://127.0.0.1:5000")


def _chrome():
    env = os.environ.get("PW_CHROME")
    if env and os.path.exists(env):
        return env
    for c in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    ]:
        if os.path.exists(c):
            return c
    return None


def main():
    with sync_playwright() as p:
        kw = {"args": ["--no-sandbox"]}
        if _chrome():
            kw["executable_path"] = _chrome()
        browser = p.chromium.launch(**kw)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(BASE + "/", wait_until="networkidle")
        # Expand everything so the layer advanced settings are in the DOM.
        page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")

        # Fast preview: keep streets + water, drop buildings/green.
        for name in ("buildings", "green"):
            el = page.query_selector(f"#{name}_enabled")
            if el and el.is_checked():
                el.click()
        page.fill("#location_distance", "1200")
        page.fill("#location_latitude", "51.8986")
        page.fill("#location_longitude", "-8.4756")

        # Sanity: the new field exists on the water layer.
        have_field = page.query_selector("#water_min_width") is not None

        def preview(min_width):
            page.fill("#water_min_width", str(min_width))
            return page.evaluate(
                """async () => {
                    const fd = new FormData(document.getElementById('map-form'));
                    fd.set('location_latitude','51.8986');
                    fd.set('location_longitude','-8.4756');
                    fd.set('location_distance','1200');
                    fd.set('location_data_source','local');
                    fd.set('location_pbf_path','../data/regions/cork.osm.pbf');
                    fd.set('water_enabled','on');
                    fd.set('streets_enabled','on');
                    if (document.querySelector('#buildings_enabled')) document.querySelector('#buildings_enabled').checked=false;
                    if (document.querySelector('#green_enabled')) document.querySelector('#green_enabled').checked=false;
                    const r = await fetch('/api/preview', {method:'POST', body: fd});
                    const ct = r.headers.get('content-type')||'';
                    const j = ct.includes('application/json') ? await r.json() : null;
                    const svg = j && j.svg ? j.svg : '';
                    const filled = (svg.match(/fill: ?#1f24bd/g) || []).length;
                    return {status: r.status, is_json: ct.includes('application/json'),
                            success: j ? j.success : null, water_fills: filled};
                }"""
            )

        r0 = preview(0)
        r20 = preview(20)

        browser.close()

    report = {
        "has_water_min_width_field": have_field,
        "preview_minwidth0": r0,
        "preview_minwidth20": r20,
        "slivers_removed": (r0.get("water_fills", 0) - r20.get("water_fills", 0))
        if r0.get("is_json") and r20.get("is_json") else None,
        "page_errors": errors[:10],
    }
    print(json.dumps(report, indent=2, default=str))

    ok = (
        have_field
        and r0.get("is_json") and r20.get("is_json")
        and r0.get("success") is True and r20.get("success") is True
        and r20.get("water_fills", 0) < r0.get("water_fills", 0)
        and not errors
    )
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
