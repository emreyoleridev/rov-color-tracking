"""Video I/O helpers: read any OpenCV-supported file, write browser-playable H.264 MP4."""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import cv2
import imageio_ffmpeg
import numpy as np


def fit_size(w: int, h: int, max_w: int = 640) -> tuple[int, int]:
    """Downscale to `max_w` and round to multiples of 16 (H.264 macroblocks)."""
    s = min(1.0, max_w / w)
    return max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)


def read_frames(path: str, max_w: int = 640, max_frames: int = 900) -> tuple[Iterator[np.ndarray], float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    size = fit_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), max_w)

    def it():
        try:
            for _ in range(max_frames):
                ok, f = cap.read()
                if not ok:
                    break
                yield cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        finally:
            cap.release()

    return it(), fps, min(n, max_frames)


def write_mp4(path: str, frames: Iterable[np.ndarray], fps: float):
    """Write BGR frames as H.264/yuv420p so the video plays in every browser."""
    writer = None
    for f in frames:
        if writer is None:
            h, w = f.shape[:2]
            writer = imageio_ffmpeg.write_frames(path, (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                                 quality=None, macro_block_size=16,
                                                 output_params=["-crf", "28", "-preset", "veryfast",
                                                                "-movflags", "+faststart"])
            writer.send(None)
        writer.send(np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    if writer is not None:
        writer.close()
