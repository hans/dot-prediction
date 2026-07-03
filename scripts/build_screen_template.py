"""Build the iPad-screen template image used for ECC alignment.

The task renders a static gradient (purp.png) onto an HTML5 canvas. The canvas
does NOT fill the whole iPad screen — it occupies a sub-rectangle determined by
the task's box_width (=100 CSS px) and screen_fill (=0.98) constants. Outside
the canvas, the page background is white (default jsPsych body).

iPad Pro 11" M2 landscape: 2388 x 1668 device px, DPR=2, CSS = 1194 x 834.
  canvas CSS  = (1194 - 2*100) * 0.98  x  834 * 0.98  =  974 x 817
  canvas dev  = 1948 x 1634
  padding dev = ~220 px left/right, ~17 px top/bottom
"""

from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parent.parent / "data"
SRC = DATA / "purp.png"  # 800x960 source, despite the extension it's JPEG

# iPad Pro 11" M2 native landscape (device pixels)
IPAD_W, IPAD_H = 2388, 1668

# Task constants
BOX_WIDTH_CSS = 100        # box_width in plugin-dot-task-ecog.js
SCREEN_FILL = 0.98
DPR = 2                    # iPad device-pixel ratio
# CSS viewport on iPad Pro 11" M2 landscape
CSS_W, CSS_H = 1194, 834


def canvas_rect_device_px():
    """Return (x0, y0, w, h) of the canvas region in iPad device pixels."""
    canvas_css_w = (CSS_W - 2 * BOX_WIDTH_CSS) * SCREEN_FILL
    canvas_css_h = CSS_H * SCREEN_FILL
    w = round(canvas_css_w * DPR)
    h = round(canvas_css_h * DPR)
    x0 = (IPAD_W - w) // 2
    y0 = (IPAD_H - h) // 2
    return x0, y0, w, h


def build_full_stretch():
    """Variant A: stretch the gradient to the full iPad screen.
    Use this if the canvas effectively fills the screen on the device."""
    src = Image.open(SRC).convert("RGB")
    out = src.resize((IPAD_W, IPAD_H), Image.BILINEAR)
    return out


def build_padded():
    """Variant B: gradient inside the canvas sub-rectangle, white elsewhere.
    Matches the actual task layout on iPad Pro 11"."""
    src = Image.open(SRC).convert("RGB")
    x0, y0, w, h = canvas_rect_device_px()
    out = Image.new("RGB", (IPAD_W, IPAD_H), (255, 255, 255))
    out.paste(src.resize((w, h), Image.BILINEAR), (x0, y0))
    return out


def main():
    rect = canvas_rect_device_px()
    print(f"Canvas region (device px): x={rect[0]} y={rect[1]} w={rect[2]} h={rect[3]}")

    full = build_full_stretch()
    padded = build_padded()

    out_full = DATA / "template_full_2388x1668.png"
    out_padded = DATA / "template_padded_2388x1668.png"
    full.save(out_full)
    padded.save(out_padded)
    print(f"wrote {out_full}")
    print(f"wrote {out_padded}")


if __name__ == "__main__":
    main()
