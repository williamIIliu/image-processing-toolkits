from __future__ import annotations

import json
import math
from pathlib import Path

import imageio
from moviepy.video.io.VideoFileClip import VideoFileClip

WORKSPACE = Path("/root/projects/TOBench/tasks/Customer_Service_LIU/Ecommence-After-sales_Complaint/initial_workspace")
VIDEO_PATH = WORKSPACE / "clip.mp4"
OUTPUT_ROOT = WORKSPACE / "output_frames"
PER_SECOND_DIR = OUTPUT_ROOT / "per_seconds"
PER_FRAME_DIR = OUTPUT_ROOT / "per_frame"

# default sampling configuration
PER_SECOND_INTERVAL = 1            # seconds between captures for per-second export
PER_FRAME_SECOND = 1               # which second to dump all frames
FRAME_AT_SECOND_TARGET = 1         # second to capture via save_frame_at_second
FRAME_BY_INDEX_TARGET = 10          # absolute frame index for save_frame_by_index
FRAME_AT_SECOND_OUTPUT = OUTPUT_ROOT / "frame_at_second.jpg"
FRAME_BY_INDEX_OUTPUT = OUTPUT_ROOT / "frame_by_index.jpg"


def get_video_metadata_json(video_clip: VideoFileClip) -> str:
    """Build video metadata as a JSON string (not saved to disk)."""
    fps = video_clip.fps or 0
    approx_total_frames = int(round(fps * video_clip.duration)) if fps else None
    data = {
        "duration_seconds": round(video_clip.duration, 2),
        "frame_size": video_clip.size,
        "fps": round(fps, 2),
        "approx_total_frames": approx_total_frames,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def export_frames_every_second(video_clip: VideoFileClip, output_dir: Path, interval_seconds: int) -> None:
    """Export one frame every `interval_seconds` and store under `per_seconds`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = math.floor(video_clip.duration)
    exported = 0

    for second in range(0, duration, interval_seconds):
        frame = video_clip.get_frame(second)
        output_path = output_dir / f"sec_{second:04d}.jpg"
        imageio.imwrite(output_path, frame)
        exported += 1
        print(f"[per_second] saved {output_path.name}")

    print(f"[per_second] exported {exported} frame(s) to {output_dir.resolve()}")


def export_frames_for_second(
    video_clip: VideoFileClip,
    start_second: int,
    output_dir: Path,
) -> None:
    """Export all frames inside the [start_second, start_second+1) interval."""
    fps = video_clip.fps or 0
    if fps <= 0:
        raise ValueError("Video FPS is unavailable; cannot pick frames within a second.")

    if start_second < 0 or start_second >= video_clip.duration:
        raise ValueError("start_second is outside the video duration.")

    interval_duration = min(1.0, video_clip.duration - start_second)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0

    max_frames_in_second = max(1, math.floor(interval_duration * fps))
    for frame_idx in range(max_frames_in_second):
        frame_timestamp = start_second + frame_idx / fps
        frame = video_clip.get_frame(frame_timestamp)
        output_path = output_dir / f"sec_{start_second:04d}_frame_{frame_idx:04d}.jpg"
        imageio.imwrite(output_path, frame)
        exported += 1
        print(
            f"[per_frame] saved {output_path.name} "
            f"(second={start_second}, frame_index={frame_idx}, timestamp={frame_timestamp:.3f}s)"
        )

    print(f"[per_frame] exported {exported} frame(s) for second {start_second}")


def export_frame_at_second(
    video_clip: VideoFileClip,
    second: int,
    output_path: Path | str,
) -> Path:
    """Save the frame at an exact second to the provided output path and return it."""
    if second < 0 or second >= video_clip.duration:
        raise ValueError("second is outside the video duration.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = video_clip.get_frame(second)
    imageio.imwrite(output_path, frame)
    print(f"[frame_at_second] saved {output_path} (second={second})")
    return output_path


def export_frame_by_index(
    video_clip: VideoFileClip,
    frame_index: int,
    output_path: Path | str,
) -> Path:
    """Save the frame by absolute frame index (0-based) to the provided output path."""
    fps = video_clip.fps or 0
    if fps <= 0:
        raise ValueError("Video FPS is unavailable; cannot pick frame by index.")
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative.")

    total_frames = int(video_clip.duration * fps)
    if frame_index >= total_frames:
        raise ValueError(f"frame_index must be < {total_frames}.")

    frame_timestamp = frame_index / fps
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = video_clip.get_frame(frame_timestamp)
    imageio.imwrite(output_path, frame)
    print(f"[frame_by_index] saved {output_path} (frame_index={frame_index}, timestamp={frame_timestamp:.3f}s)")
    return output_path


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found at {VIDEO_PATH}")

    with VideoFileClip(str(VIDEO_PATH)) as clip:
        metadata_json = get_video_metadata_json(clip)
        print(f"[metadata]\n{metadata_json}")

        export_frames_every_second(
            video_clip=clip,
            output_dir=PER_SECOND_DIR,
            interval_seconds=PER_SECOND_INTERVAL,
        )

        export_frames_for_second(
            video_clip=clip,
            start_second=PER_FRAME_SECOND,
            output_dir=PER_FRAME_DIR,
        )

        export_frame_at_second(
            video_clip=clip,
            second=FRAME_AT_SECOND_TARGET,
            output_path=FRAME_AT_SECOND_OUTPUT,
        )

        export_frame_by_index(
            video_clip=clip,
            frame_index=FRAME_BY_INDEX_TARGET,
            output_path=FRAME_BY_INDEX_OUTPUT,
        )


if __name__ == "__main__":
    main()