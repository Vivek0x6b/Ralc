"""Add a section-synced offline voiceover to docs/demo.mp4 (pyttsx3, no API).

The narration is split so each sentence group starts exactly when its matching
caption card appears in the video. Those timestamps are recomputed by replaying
the render timing in docs/make_demo_video.py (same FPS and hold durations). The
video stays 48s unchanged; only the audio track is rebuilt from the silent cut.

Groups (caption -> narration):
  Same question, same 1,000-token budget  -> setup, through the question/budget
  Vector search drops the AWS message      -> the vector sentence
  Gemma found: the AWS message replaces... -> the Gemma sentence
  RALC keeps both: the full story          -> the RALC sentence

    python docs/add_voiceover.py
"""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import pyttsx3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEMO = HERE / "demo.mp4"
SILENT = HERE / "demo_silent.mp4"
RATE = 165
FPS = 24   # must match make_demo_video.py

# caption index -> the narration that starts when that card appears
GROUPS = [
    (0, "An AI model only knows the text you send it. In a long chat, something "
        "has to choose what goes in. Here is one question, with a one thousand "
        "token budget."),
    (1, "Vector search finds the Heroku message, but drops the one about "
        "switching to AWS, because the two look alike."),
    (2, "Gemma noticed that the AWS message replaces the Heroku one."),
    (3, "So RALC keeps both, and the model gets the full story."),
]


def _frames(seconds: float) -> int:
    return max(1, int(seconds * FPS))


def caption_start_seconds() -> list[float]:
    """Replay make_demo_video's timing to find when each caption card starts."""
    proc = subprocess.run([sys.executable, "-m", "examples.demo"], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    lines = proc.stdout.rstrip("\n").split("\n")

    def find(prefix):
        return next(i for i, ln in enumerate(lines) if ln.strip().startswith(prefix))
    i1, i2, i3, i4 = find("1)"), find("2)"), find("3)"), find("SUMMARY")
    sections = [lines[:i1 - 1], lines[i1 - 1:i2 - 1], lines[i2 - 1:i3 - 1],
                lines[i3 - 1:i4 - 1]]

    starts, t = [], 0
    for seg in sections:
        starts.append(t / FPS)          # caption card begins here
        t += _frames(3.0)               # caption card hold
        t += len(seg) * _frames(0.55)   # line-by-line reveal
        t += _frames(1.3)               # section end hold
    return starts


RATE_MAX = 255


def say(text: str, path: Path, rate: int) -> float:
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.save_to_file(text, str(path))
    engine.runAndWait()
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def video_seconds(path: Path) -> float:
    reader = imageio.get_reader(path)
    dur = reader.get_meta_data().get("duration")
    reader.close()
    return float(dur)


def main():
    if not SILENT.exists():
        raise SystemExit("docs/demo_silent.mp4 is missing; render the silent video first.")
    video_len = video_seconds(SILENT)
    starts = caption_start_seconds()
    # Window each group has before the next caption (the last runs to the end).
    windows = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)] + \
              [video_len - starts[-1]]

    clips = []
    for list_idx, (cap_idx, text) in enumerate(GROUPS):
        wav = HERE / f"_vo_{cap_idx}.wav"
        window = windows[cap_idx]
        dur = say(text, wav, RATE)
        # If the narration would run past the next caption, speed up just this
        # group so it fits, capped so it stays intelligible.
        if dur > window * 0.92:
            faster = min(RATE_MAX, int(RATE * dur / (window * 0.9)) + 1)
            dur = say(text, wav, faster)
            print(f"  caption {cap_idx}: sped to rate {faster} to fit {window:.2f}s", flush=True)
        clips.append((wav, starts[cap_idx], dur, window))
        print(f"  caption {cap_idx}: start {starts[cap_idx]:.2f}s, narration {dur:.2f}s, "
              f"window {window:.2f}s, ends {starts[cap_idx] + dur:.2f}s", flush=True)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", str(SILENT)]
    for wav, _, _, _ in clips:
        cmd += ["-i", str(wav)]
    parts = []
    for n, (_, start, _, window) in enumerate(clips, start=1):
        # Safety trim so a group can never bleed into the next caption.
        keep = max(0.5, window * 0.98)
        parts.append(f"[{n}:a]atrim=0:{keep:.3f},adelay={int(start * 1000)}:all=1[a{n}]")
    mixin = "".join(f"[a{n}]" for n in range(1, len(clips) + 1))
    parts.append(f"{mixin}amix=inputs={len(clips)}:normalize=0:dropout_transition=0[m]")
    parts.append("[m]apad[a]")
    cmd += ["-filter_complex", ";".join(parts),
            "-map", "0:v", "-map", "[a]", "-t", f"{video_len:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", str(DEMO)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for wav, _, _, _ in clips:
        wav.unlink(missing_ok=True)
    print(f"wrote {DEMO.name} ({video_seconds(DEMO):.2f}s, voiceover synced to captions); "
          f"kept {SILENT.name}")


if __name__ == "__main__":
    main()
