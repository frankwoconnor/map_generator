"""Headless Playwright smoke test for the running Flask studio.

Drives system Chrome (Chromium download was unavailable) against
http://127.0.0.1:5000/, capturing console errors, page errors, failed
requests, and asserting key UI sections render and APIs respond.
"""
import sys
import json
import shutil
import os
import urllib.request

from playwright.sync_api import sync_playwright

BASE = os.environ.get("PW_BASE", "http://127.0.0.1:5000")


def _find_chrome():
    """Locate a Chrome/Chromium executable across platforms/env."""
    env = os.environ.get("PW_CHROME")
    if env and os.path.exists(env):
        return env
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None  # fall back to Playwright's bundled chromium if present


CHROME = _find_chrome()

console_errors = []
page_errors = []
failed_requests = []


def main():
    with sync_playwright() as p:
        launch_kwargs = {"args": ["--no-sandbox"]}
        if CHROME:
            launch_kwargs["executable_path"] = CHROME
        browser = p.chromium.launch(**launch_kwargs)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})

        page.on("console", lambda m: console_errors.append(f"{m.type}: {m.text}")
                if m.type in ("error",) else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("requestfailed", lambda r: failed_requests.append(
            f"{r.method} {r.url} -> {r.failure}"))

        results = {}

        # 1. Load home page
        resp = page.goto(BASE + "/", wait_until="networkidle")
        results["home_status"] = resp.status
        results["home_has_form"] = page.locator("#map-form").count() == 1

        # 2. Key UI sections present
        for sel in ["#generate-btn", "#preset-gallery", "#palette-gallery",
                    "text=Output Formats", "text=SVG Shape Simplification",
                    "text=Quick toggle", "#map-form"]:
            results[f"present:{sel}"] = page.locator(sel).count() > 0

        # 3. JSON APIs respond
        for path in ["/api/style", "/api/runs?limit=5", "/api/palettes",
                     "/api/presets", "/api/regions"]:
            try:
                r = urllib.request.urlopen(BASE + path, timeout=10)
                data = json.loads(r.read().decode())
                results[f"api:{path}"] = data.get("success", "ok")
            except Exception as e:
                results[f"api:{path}"] = f"ERR {e}"

        # 4. Verify /api/style builds buildings layer (the crash path)
        r = urllib.request.urlopen(BASE + "/api/style", timeout=10)
        style = json.loads(r.read().decode())["style"]
        results["style_buildings_has_filters"] = "filters" in style["layers"]["buildings"]
        results["style_layers"] = sorted(style["layers"].keys())

        # 5. Interact: expand all <details> (so hidden sections render), no crash
        page.evaluate(
            "document.querySelectorAll('details').forEach(d => d.open = true)")
        page.wait_for_timeout(500)
        results["after_expand_pageerror_count"] = len(page_errors)

        # 6. Click the mask select to ensure creative controls wire up
        if page.locator("#output_mask").count():
            page.select_option("#output_mask", "circle")
            page.wait_for_timeout(200)
            results["mask_select_ok"] = True

        results["console_errors"] = console_errors[:20]
        results["page_errors"] = page_errors[:20]
        results["failed_requests"] = failed_requests[:20]

        browser.close()

    print(json.dumps(results, indent=2, default=str))
    # Non-zero exit if there are blocking errors
    if page_errors or any("ERR" in str(v) for v in results.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
