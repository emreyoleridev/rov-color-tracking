"""Color-based target tracking: HSV segmentation -> morphology -> contours -> guidance command.

All functions work on 8-bit BGR frames (OpenCV convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

# HSV ranges in OpenCV units (H: 0-179, S/V: 0-255). If h_low > h_high the hue range wraps around 0 (red).
PRESETS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "Orange buoy": ((5, 80, 60), (22, 255, 255)),
    "Red": ((170, 120, 70), (8, 255, 255)),
    "Yellow": ((22, 110, 110), (35, 255, 255)),
    "Green": ((40, 90, 60), (85, 255, 255)),
    "Blue": ((95, 120, 60), (130, 255, 255)),
    "Magenta": ((140, 90, 70), (170, 255, 255)),
}


@dataclass
class TrackerConfig:
    lower: tuple[int, int, int] = PRESETS["Orange buoy"][0]
    upper: tuple[int, int, int] = PRESETS["Orange buoy"][1]
    blur: int = 5                 # Gaussian blur kernel before HSV conversion (0 = off)
    open_kernel: int = 5          # opening removes speckle noise
    close_kernel: int = 9         # closing fills holes from specular highlights
    iterations: int = 1
    min_area: float = 150.0       # contours smaller than this (px) are ignored
    deadzone: float = 0.10        # |offset| below this fraction of half-width/height counts as centered
    stop_area: float = 0.08       # target area / frame area at which the vehicle stops approaching
    smoothing: float = 0.5        # EMA weight of the new measurement (1 = no smoothing)
    max_lost: int = 10            # frames to coast on the last estimate before declaring the target lost
    morphology: bool = True


@dataclass
class Detection:
    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]
    radius: float
    circularity: float
    contour: np.ndarray = field(repr=False)


@dataclass
class Command:
    dx: float          # normalized horizontal offset, -1 (left edge) .. +1 (right edge)
    dy: float          # normalized vertical offset, -1 (top) .. +1 (bottom)
    yaw: str           # LEFT / RIGHT / CENTER
    heave: str         # UP / DOWN / LEVEL
    surge: str         # FORWARD / STOP
    yaw_rate: float    # proportional controller output, -1..1
    heave_rate: float
    state: str         # TRACKING / COASTING / SEARCHING

    @property
    def text(self) -> str:
        if self.state == "SEARCHING":
            return "SEARCH (rotate to find target)"
        parts = [p for p in (self.yaw if self.yaw != "CENTER" else "",
                             self.heave if self.heave != "LEVEL" else "") if p]
        parts.append(self.surge)
        return " + ".join(parts)


def _odd(k: int) -> int:
    return k if k % 2 else k + 1


def segment(frame: np.ndarray, cfg: TrackerConfig) -> np.ndarray:
    """Binary mask of pixels inside the HSV range (handles hue wrap-around for red)."""
    if cfg.blur > 1:
        frame = cv2.GaussianBlur(frame, (_odd(cfg.blur), _odd(cfg.blur)), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(cfg.lower, np.uint8), np.array(cfg.upper, np.uint8)
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    a = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]], np.uint8))
    b = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi)
    return cv2.bitwise_or(a, b)


def clean_mask(mask: np.ndarray, cfg: TrackerConfig) -> np.ndarray:
    """Opening removes isolated noise pixels, closing fills holes inside the target."""
    if not cfg.morphology:
        return mask
    if cfg.open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=cfg.iterations)
    if cfg.close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=cfg.iterations)
    return mask


def find_target(mask: np.ndarray, cfg: TrackerConfig) -> tuple[Detection | None, list[np.ndarray]]:
    """Largest external contour above `min_area`; returns (best, all valid candidates)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= cfg.min_area]
    if not valid:
        return None, []
    c = max(valid, key=cv2.contourArea)
    area = cv2.contourArea(c)
    m = cv2.moments(c)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    perim = cv2.arcLength(c, True)
    (_, _), radius = cv2.minEnclosingCircle(c)
    return Detection(cx, cy, area, cv2.boundingRect(c), radius,
                     4 * np.pi * area / (perim * perim) if perim else 0.0, c), valid


def compute_command(cx: float, cy: float, area: float, shape: tuple[int, ...], cfg: TrackerConfig,
                    state: str = "TRACKING") -> Command:
    """Offset of the target from the image center -> yaw / heave / surge command."""
    h, w = shape[:2]
    dx = (cx - w / 2) / (w / 2)
    dy = (cy - h / 2) / (h / 2)
    yaw = "CENTER" if abs(dx) < cfg.deadzone else ("RIGHT" if dx > 0 else "LEFT")
    heave = "LEVEL" if abs(dy) < cfg.deadzone else ("DOWN" if dy > 0 else "UP")
    surge = "STOP" if area / (w * h) >= cfg.stop_area else "FORWARD"
    rate = lambda d: 0.0 if abs(d) < cfg.deadzone else float(np.clip(d, -1, 1))  # noqa: E731
    return Command(dx, dy, yaw, heave, surge, rate(dx), rate(dy), state)


SEARCH = Command(0.0, 0.0, "CENTER", "LEVEL", "STOP", 0.0, 0.0, "SEARCHING")


@dataclass
class FrameResult:
    mask_raw: np.ndarray
    mask: np.ndarray
    detection: Detection | None
    candidates: list[np.ndarray]
    command: Command
    estimate: tuple[float, float, float] | None   # smoothed (cx, cy, area)


class ColorTracker:
    """Stateful tracker: per-frame detection + exponential smoothing + short coasting when the target is lost."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.est: np.ndarray | None = None
        self.lost = 0

    def reset(self):
        self.est, self.lost = None, 0

    def update(self, frame: np.ndarray) -> FrameResult:
        cfg = self.cfg
        raw = segment(frame, cfg)
        mask = clean_mask(raw, cfg)
        det, cands = find_target(mask, cfg)
        if det is not None:
            z = np.array([det.cx, det.cy, det.area])
            self.est = z if self.est is None else cfg.smoothing * z + (1 - cfg.smoothing) * self.est
            self.lost, state = 0, "TRACKING"
        else:
            self.lost += 1
            if self.lost > cfg.max_lost:
                self.est = None
            state = "COASTING"
        if self.est is None:
            cmd = SEARCH
        else:
            cmd = compute_command(*self.est, frame.shape, cfg, state)
        est = None if self.est is None else tuple(map(float, self.est))
        return FrameResult(raw, mask, det, cands, cmd, est)


def process_image(frame: np.ndarray, cfg: TrackerConfig) -> FrameResult:
    """Single-frame (stateless) tracking."""
    return ColorTracker(replace(cfg, smoothing=1.0)).update(frame)


# --- Visualization -----------------------------------------------------------------------------

_GREEN, _CYAN, _YELLOW, _RED, _WHITE = (80, 220, 80), (230, 200, 40), (40, 220, 250), (60, 60, 240), (255, 255, 255)


def _label(img, text, org, color=_WHITE, scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, res: FrameResult, cfg: TrackerConfig) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    c = (w // 2, h // 2)
    # dead-zone box and crosshair
    dzx, dzy = int(cfg.deadzone * w / 2), int(cfg.deadzone * h / 2)
    cv2.rectangle(out, (c[0] - dzx, c[1] - dzy), (c[0] + dzx, c[1] + dzy), _CYAN, 1, cv2.LINE_AA)
    cv2.line(out, (c[0] - 12, c[1]), (c[0] + 12, c[1]), _CYAN, 1, cv2.LINE_AA)
    cv2.line(out, (c[0], c[1] - 12), (c[0], c[1] + 12), _CYAN, 1, cv2.LINE_AA)
    for cand in res.candidates:
        if res.detection is None or cand is not res.detection.contour:
            cv2.drawContours(out, [cand], -1, _YELLOW, 1, cv2.LINE_AA)
    if res.detection is not None:
        d = res.detection
        cv2.drawContours(out, [d.contour], -1, _GREEN, 2, cv2.LINE_AA)
        x, y, bw, bh = d.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), _GREEN, 1, cv2.LINE_AA)
    if res.estimate is not None:
        ex, ey = int(res.estimate[0]), int(res.estimate[1])
        cv2.circle(out, (ex, ey), 5, _RED, -1, cv2.LINE_AA)
        cv2.arrowedLine(out, c, (ex, ey), _RED, 2, cv2.LINE_AA, tipLength=0.08)
    cmd = res.command
    cv2.rectangle(out, (0, 0), (w, 54), (0, 0, 0), -1)
    color = {"TRACKING": _GREEN, "COASTING": _YELLOW, "SEARCHING": _RED}[cmd.state]
    _label(out, f"{cmd.state}", (10, 20), color)
    _label(out, f"CMD: {cmd.text}", (10, 44), _WHITE)
    if cmd.state != "SEARCHING":
        _label(out, f"dx={cmd.dx:+.2f}  dy={cmd.dy:+.2f}", (w - 190, 20), _WHITE, 0.5)
    return out
