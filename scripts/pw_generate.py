"""Playwright end-to-end generation test: click Generate, watch the UI.

Drives system Chrome against the running studio, clicks #generate-btn with the
form's default (valid, local-cork) location, and samples the job panel,
status indicator, live log, and result/download wiring over time. Captures
console + page errors (e.g. the JSON.parse class) and the /api/generate,
/api/jobs/<id> and SSE responses so missing feedback can be pinpointed.

Env: PW_BASE (default http://127.0.0.1:5000), PW_CHROME, PW_TIMEOUT (sec, 120).
"""
import os
import json
import sys
import time

from playwright.sync_api import sync_playwright

BASE = os.environ.get("PW_BASE", "http://127.0.0.1:5000")
TIMEOUT = float(os.environ.get("PW_TIMEOUT", "120"))


def _find_chrome():
    env = os.environ.get("PW_CHROME")
    if env and os.path.exists(env):
        return env
    for c in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome", "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]:
        if os.path.exists(c):
            return c
    return None


def main():
    trace = {"api_responses": [], "console_errors": [], "page_errors": []}
    with sync_playwright() as p:
        kw = {"args": ["--no-sandbox"]}
        if _find_chrome():
            kw["executable_path"] = _find_chrome()
        browser = p.chromium.launch(**kw)
        page = browser.new_page(viewport={"width": 1400, "height": 1100})

        page.on("console", lambda m: trace["console_errors"].append(
            f"{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: trace["page_errors"].append(str(e)))

        def on_response(resp):
            u = resp.url
            if "/api/generate" in u or "/api/jobs/" in u:
                ct = resp.headers.get("content-type", "")
                trace["api_responses"].append(
                    {"url": u.split("5000")[-1], "status": resp.status, "ct": ct})

        page.on("response", on_response)

        page.goto(BASE + "/", wait_until="networkidle")

        # Speed up the run: use a smaller radius over the local cork PBF.
        page.fill("#location_distance", "1500")
        page.fill("#location_latitude", "51.8986")
        page.fill("#location_longitude", "-8.4756")

        # Ensure the result panel is in a clean state to start
        pre = page.locator("#job-panel").get_attribute("hidden")
        # Click Generate (form defaults are a valid local-cork location)
        page.click("#generate-btn")

        # Sample the UI over time
        samples = []
        t0 = time.time()
        final = None
        started = False
        while time.time() - t0 < TIMEOUT:
            snap = page.evaluate(
                """() => {
                    const q = s => document.querySelector(s);
                    const panel = q('#job-panel');
                    const label = q('#job-status-label');
                    const status = q('#status-indicator');
                    const log = q('#job-log');
                    const dl = q('#download-svg');
                    const img = q('#preview-image');
                    return {
                        panel_visible: panel ? !panel.hidden : null,
                        job_label: label ? label.textContent.trim() : null,
                        status: status ? status.textContent.trim() : null,
                        log_lines: log ? log.textContent.split('\\n').filter(Boolean).length : 0,
                        log_tail: log ? log.textContent.split('\\n').filter(Boolean).slice(-2) : [],
                        dl_href: dl ? dl.getAttribute('href') : null,
                        img_src: img ? img.getAttribute('src') : null,
                        gen_disabled: q('#generate-btn') ? q('#generate-btn').disabled : null,
                    };
                }"""
            )
            samples.append({"t": round(time.time() - t0, 1), **snap})
            started = started or snap["panel_visible"] or (
                snap["status"] not in (None, "", "Ready"))
            # Only consider ourselves finished once the job actually started and
            # then reached a terminal UI state.
            if started and (
                snap["status"] in ("Done", "Error", "Cancelled")
                or (snap["gen_disabled"] is False and snap["log_lines"] > 0)
            ):
                final = snap
                break
            time.sleep(1.5)

        browser.close()

    report = {
        "api_responses": trace["api_responses"],
        "panel_visible_before_click": pre is None,  # hidden attr absent -> visible
        "started": started,
        "samples_head": samples[:4],
        "samples_tail": samples[-4:],
        "num_samples": len(samples),
        "final_status": (final or {}).get("status"),
        "final_job_label": (final or {}).get("job_label"),
        "final_log_lines": (final or {}).get("log_lines"),
        "final_dl_href": (final or {}).get("dl_href"),
        "console_errors": trace["console_errors"][:20],
        "page_errors": trace["page_errors"][:20],
    }
    print(json.dumps(report, indent=2, default=str))

    ok = (final is not None
          and (final.get("status") == "Done")
          and final.get("panel_visible")
          and final.get("log_lines", 0) > 0
          and not trace["page_errors"])
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
