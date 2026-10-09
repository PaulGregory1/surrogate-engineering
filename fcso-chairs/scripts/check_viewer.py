"""Headless check of report/viewer.html (playwright driving the installed Google Chrome, file:// URL).

    ./.venv/bin/python scripts/check_viewer.py [--chair <id prefix>] [--others] [--out report/viewer_check]

Opens the page by file URL (as a double-click would), waits for three.js and the chair's meshes, drags one
cell to rotate the shared camera, and saves screenshots before/after the drag. Fails (exit 1) on any console
error, page error or failed request, or if no 3D cell was built.
"""
import argparse
import os
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chair", default="")
    ap.add_argument("--others", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "viewer_check"))
    ap.add_argument("--width", type=int, default=1500)
    a = ap.parse_args()
    url = "file://" + os.path.join(ROOT, "report", "viewer.html") + (f"#{a.chair}" if a.chair else "")
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=["--use-angle=metal", "--enable-webgl",
                                                                     "--ignore-gpu-blocklist"])
        page = b.new_page(viewport=dict(width=a.width, height=1000), device_scale_factor=1)
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: errors.append(f"requestfailed: {r.url} {r.failure}"))
        page.goto(url)
        page.wait_for_function("window.THREE && window.FCSO_MESHES && Object.keys(window.FCSO_MESHES).length > 0",
                               timeout=60000)
        if a.others:
            page.check("#others")
        page.wait_for_timeout(1500)
        n_views = page.evaluate("document.querySelectorAll('.view:not(.empty)').length")
        n_empty = page.evaluate("document.querySelectorAll('.view.empty').length")
        page.screenshot(path=a.out + "_1.png", full_page=False)
        box = page.locator(".view:not(.empty)").first.bounding_box()
        cam0 = page.evaluate("[camera.position.x, camera.position.y, camera.position.z]")
        if box:
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 60, y + 15, steps=8)
            page.mouse.up()
            page.wait_for_timeout(500)
        cam1 = page.evaluate("[camera.position.x, camera.position.y, camera.position.z]")
        page.screenshot(path=a.out + "_2.png", full_page=False)
        b.close()
    print(f"views {n_views}, empty cells {n_empty}, camera {cam0} -> {cam1}")
    real = [e for e in errors if "GPU stall due to ReadPixels" not in e and "GroupMarkerNotSet" not in e]
    for e in real:
        print(e)
    ok = n_views > 0 and not real and cam0 != cam1
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
