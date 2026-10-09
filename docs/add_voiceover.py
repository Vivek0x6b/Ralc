"""Add an offline voiceover to docs/demo.mp4 (pyttsx3, Windows SAPI5, no API).

Generates the narration with the built-in Windows voice, keeps the silent cut as
docs/demo_silent.mp4, and muxes the audio in with the ffmpeg binary that ships
with imageio-ffmpeg. The shorter track is padded so the final video and audio
end together (freeze the last video frame if the audio is longer, pad the audio
with silence if the video is longer).

    python docs/add_voiceover.py
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import pyttsx3

HERE = Path(__file__).resolve().parent
DEMO = HERE / "demo.mp4"
SILENT = HERE / "demo_silent.mp4"
WAV = HERE / "_narration.wav"

NARRATION = (
    "An AI model only knows the text you send it. In a long chat, something has "
    "to choose what goes in. Here is one question, with a one thousand token "
    "budget. Vector search finds the Heroku message, but drops the one about "
    "switching to AWS, because the two look alike. Gemma noticed that the AWS "
    "message replaces the Heroku one. So RALC keeps both, and the model gets the "
    "full story."
)


def make_narration(path: Path) -> None:
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.save_to_file(NARRATION, str(path))
    engine.runAndWait()


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def video_seconds(path: Path) -> float:
    reader = imageio.get_reader(path)
    dur = reader.get_meta_data().get("duration")
    reader.close()
    return float(dur)


def main():
    print("generating narration (offline Windows voice) ...", flush=True)
    make_narration(WAV)
    audio_len = wav_seconds(WAV)

    # Preserve the current (silent) video before overwriting demo.mp4.
    shutil.copyfile(DEMO, SILENT)
    video_len = video_seconds(SILENT)

    target = max(video_len, audio_len)
    pad_v = max(0.0, target - video_len)
    print(f"audio {audio_len:.2f}s | video {video_len:.2f}s | target {target:.2f}s "
          f"| video pad {pad_v:.2f}s", flush=True)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-i", str(SILENT), "-i", str(WAV),
        "-filter_complex",
        f"[0:v]tpad=stop_mode=clone:stop_duration={pad_v:.3f}[v];[1:a]apad[a]",
        "-map", "[v]", "-map", "[a]", "-t", f"{target:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", str(DEMO),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    final_len = video_seconds(DEMO)
    print(f"wrote {DEMO.name} ({final_len:.2f}s with voiceover) and kept {SILENT.name}")


if __name__ == "__main__":
    main()
