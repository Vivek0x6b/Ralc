"""Render examples/demo.py's real output into docs/demo.mp4.

Runs the demo, captures its exact stdout (unedited), and renders a 1280x720
dark terminal video: caption cards between sections and the output lines
appearing one by one. No API calls. Requires Pillow and imageio[ffmpeg].

    python docs/make_demo_video.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo.mp4"

W, H = 1280, 720
FPS = 24
BG = (13, 17, 23)
FG = (208, 214, 221)
DIM = (120, 132, 144)
GREEN = (76, 194, 185)
RED = (224, 110, 90)
ACCENT = (76, 194, 185)

MARGIN_X, MARGIN_Y, LINE_H, FSIZE = 44, 34, 27, 20

BODY = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", FSIZE)
BOLD = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", FSIZE)
CAPTION = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 40)
CAPSMALL = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 22)

CAPTIONS = [
    "Same question, same 1,000-token budget",
    "Vector search drops the AWS message",
    "Gemma found: the AWS message replaces the Heroku one",
    "RALC keeps both: the full story",
]


def capture_output() -> list[str]:
    proc = subprocess.run([sys.executable, "-m", "examples.demo"], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return proc.stdout.rstrip("\n").split("\n")


def split_sections(lines: list[str]):
    def find(prefix):
        return next(i for i, ln in enumerate(lines) if ln.strip().startswith(prefix))
    i1, i2, i3, i4 = find("1)"), find("2)"), find("3)"), find("SUMMARY")
    return [lines[:i1 - 1], lines[i1 - 1:i2 - 1], lines[i2 - 1:i3 - 1],
            lines[i3 - 1:i4 - 1], lines[i4 - 1:]]


def color_for(line: str):
    s = line.strip()
    if "<=" in line:
        return GREEN
    if "required found 2/2" in line:
        return GREEN
    if "required found 1/2" in line:
        return RED
    if s.startswith(("=", "-")) or s[:2] in ("1)", "2)", "3)") or s == "SUMMARY":
        return ACCENT
    return FG


def terminal_frame(shown: list[str]) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((MARGIN_X, 14), "ralc-demo", font=CAPSMALL, fill=DIM)
    y = MARGIN_Y + 24
    for ln in shown:
        d.text((MARGIN_X, y), ln, font=BODY, fill=color_for(ln))
        y += LINE_H
    return np.asarray(img)


def wrap(text, font, max_w, draw):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def caption_frame(text: str) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    lines = wrap(text, CAPTION, W - 220, d)
    total = len(lines) * 54
    y = (H - total) // 2
    d.line([(W // 2 - 30, y - 28), (W // 2 + 30, y - 28)], fill=ACCENT, width=3)
    for ln in lines:
        w = d.textlength(ln, font=CAPTION)
        d.text(((W - w) // 2, y), ln, font=CAPTION, fill=FG)
        y += 54
    return img_to_np(img)


def img_to_np(img):
    return np.asarray(img)


def main():
    lines = capture_output()
    sections = split_sections(lines)

    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=None)

    def hold(frame, seconds):
        for _ in range(max(1, int(seconds * FPS))):
            writer.append_data(frame)

    def type_section(seg):
        shown = []
        for ln in seg:
            shown.append(ln)
            hold(terminal_frame(shown), 0.55)
        hold(terminal_frame(shown), 1.3)

    # Caption, then section, for the first four groups; summary closes it out.
    order = [(CAPTIONS[0], sections[0]), (CAPTIONS[1], sections[1]),
             (CAPTIONS[2], sections[2]), (CAPTIONS[3], sections[3]), (None, sections[4])]
    for caption, seg in order:
        if caption is not None:
            hold(caption_frame(caption), 3.0)
        type_section(seg)
    hold(terminal_frame(sections[4]), 2.5)

    writer.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
