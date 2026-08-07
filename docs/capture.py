"""Screenshot the Flow Lab and the copilot for the README, against a running platform.

Start both servers first (see engine-copilot/README.md), then:

    engine-copilot/.venv/bin/python docs/capture.py            # both
    engine-copilot/.venv/bin/python docs/capture.py --copilot  # just the sidebar

It drives the app the way a visitor would — optimise the shape, verify the optimum in
the true LBM solver, ask the agent a question — so every number in `flow-lab.png`,
`optimizer.png` and `copilot.png` is a real run, not a mock-up. Expect a couple of
minutes: the verification is an actual solve and the agent turn is an actual model call.

The viewport is made tall enough to hold the whole page rather than using `full_page`:
`main` is its own scroll container, so a full-page shot would capture only the header.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
URL = "http://localhost:5173/flow"
VIEWPORT = {"width": 1680, "height": 1760}
WIDTH = 1600  # committed width; 2x source is downsampled to keep the repo light

# Short enough that the panel is mostly transcript rather than empty paper below it.
COPILOT_VIEWPORT = {"width": 1680, "height": 600}
COPILOT_WIDTH = 760
QUESTION = "What is the Cd of this shape at Re = 130, and am I still inside the training range?"


def _fit(path: Path, width: int = WIDTH) -> None:
    im = Image.open(path).convert("RGB")
    im.resize((width, round(im.height * width / im.width)), Image.LANCZOS).save(
        path, optimize=True
    )
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")


def copilot() -> None:
    """One real agent turn: a question, the tool call it makes, the answer it writes."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(viewport=COPILOT_VIEWPORT, device_scale_factor=2).new_page()
        page.add_init_script("localStorage.setItem('engine-copilot.chat.open', '1')")
        page.goto(URL, wait_until="networkidle")
        time.sleep(7)

        # Start from an empty conversation: the agent resumes its SDK session across
        # page loads, and a transcript that opens mid-thought reads as a bug.
        page.request.post("http://localhost:8000/api/agent/reset")

        box = page.locator("textarea")
        box.click()
        box.fill(QUESTION)
        box.press("Enter")

        send = page.locator('button[aria-label="Send"]')
        for _ in range(600):  # the composer disables while the turn streams
            if send.is_disabled():
                break
            time.sleep(0.2)
        for _ in range(1200):
            if not send.is_disabled():
                break
            time.sleep(0.4)
        time.sleep(3)

        page.locator("aside").last.screenshot(path=str(HERE / "copilot.png"))
        browser.close()

    _fit(HERE / "copilot.png", COPILOT_WIDTH)


def lab() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(viewport=VIEWPORT, device_scale_factor=2).new_page()
        # The copilot panel is collapsed: the shot is about the lab. The key is the
        # provider's own '1'/'0', not a JSON boolean — anything else reads as "no
        # explicit choice" and the panel opens by itself above 1024px.
        page.add_init_script("localStorage.setItem('engine-copilot.chat.open', '0')")
        page.goto(URL, wait_until="networkidle")
        time.sleep(7)  # debounced predict + entrance animations

        page.get_by_role("button", name="Optimise").click()
        page.wait_for_selector("text=OPTIMISED CD", timeout=180_000)
        time.sleep(2)
        page.get_by_role("button", name="Verify with solver").click()
        # A regex, not `text=gap`: bare substring matching would also hit the word
        # elsewhere in the panel; only "gap 8.9 %" means the solver has answered.
        page.wait_for_selector("text=/gap [0-9]/", timeout=300_000)
        time.sleep(4)

        page.screenshot(path=str(HERE / "flow-lab.png"))
        page.locator("text=Shape optimisation").locator(
            "xpath=ancestor::div[contains(@class,'rounded-xl')][1]"
        ).screenshot(path=str(HERE / "optimizer.png"))
        browser.close()

    _fit(HERE / "flow-lab.png")
    _fit(HERE / "optimizer.png")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    if not only or "--lab" in only:
        lab()
    if not only or "--copilot" in only:
        copilot()
