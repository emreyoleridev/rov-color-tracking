"""Synthetic underwater video with a colored buoy and per-frame ground truth.

Lets the tracker be tested and scored without a real ROV: the target follows a known path, so detection rate,
centroid error and command accuracy can be measured exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Scenario:
    name: str
    turbidity: float = 0.15     # 0..1 haze blended toward the water color
    noise: float = 4.0          # Gaussian sensor noise (std, gray levels)
    distractors: int = 0        # small orange "fish" that should be rejected by min_area
    occlusion: bool = False     # target passes behind a rock
    lighting: float = 0.0       # amplitude of global brightness flicker
    blur: float = 0.0           # motion / defocus blur sigma
    particles: int = 0          # reddish suspended sediment specks (salt-like color noise)


SCENARIOS = {
    "clear": Scenario("clear"),
    "turbid": Scenario("turbid", turbidity=0.45, noise=6),
    "noisy": Scenario("noisy", noise=18, lighting=0.25, particles=500),
    "distractors": Scenario("distractors", distractors=6, noise=6),
    "occlusion": Scenario("occlusion", occlusion=True, noise=6),
    "hard": Scenario("hard", turbidity=0.4, noise=14, distractors=5, occlusion=True, lighting=0.2, blur=1.2,
                     particles=400),
}


def _background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    y = np.linspace(0, 1, h)[:, None, None]
    top, bottom = np.array([150, 120, 30]), np.array([70, 55, 10])   # BGR: teal surface -> dark depth
    bg = (top * (1 - y) + bottom * y) * np.ones((1, w, 1))
    # seabed and a few low-frequency blotches
    blot = cv2.GaussianBlur(rng.normal(0, 1, (h // 8, w // 8)), (0, 0), 3)
    blot = cv2.resize(blot, (w, h))[..., None]
    bg = bg + 18 * blot
    seabed = np.clip((np.arange(h) - 0.82 * h) / (0.18 * h), 0, 1)[:, None, None]
    bg = bg * (1 - seabed) + np.array([60, 85, 90]) * seabed
    return bg.astype(np.float32)


def _sphere(canvas: np.ndarray, cx: float, cy: float, r: float, color=(20, 110, 245)):
    """Shaded sphere (fake Lambertian + specular highlight)."""
    h, w = canvas.shape[:2]
    x0, x1 = int(max(cx - r - 2, 0)), int(min(cx + r + 3, w))
    y0, y1 = int(max(cy - r - 2, 0)), int(min(cy + r + 3, h))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    nx, ny = (xx - cx) / r, (yy - cy) / r
    d2 = nx ** 2 + ny ** 2
    inside = np.clip((1.0 - np.sqrt(d2)) * r, 0, 1)[..., None]           # anti-aliased edge
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    shade = np.clip(0.35 + 0.75 * (-0.4 * nx - 0.5 * ny + 0.77 * nz), 0.25, 1.1)[..., None]
    spec = (np.clip(-0.4 * nx - 0.5 * ny + 0.77 * nz, 0, 1) ** 40)[..., None] * 160
    col = np.array(color, np.float32) * shade + spec
    patch = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = patch * (1 - inside) + col * inside


def generate(scn: Scenario, n_frames: int = 240, size: tuple[int, int] = (640, 480), seed: int = 0):
    """Yield (frame_bgr_uint8, gt) where gt = dict(cx, cy, r, visible, mask) and mask is the visible buoy area."""
    w, h = size
    rng = np.random.default_rng(seed)
    bg = _background(w, h, rng)
    water = np.array([140, 125, 60], np.float32)
    rock = np.array([[0.30 * w, h], [0.36 * w, 0.42 * h], [0.47 * w, 0.30 * h], [0.58 * w, 0.40 * h],
                     [0.63 * w, h]], np.int32)
    fish = [dict(x=rng.uniform(0, w), y=rng.uniform(0.15 * h, 0.8 * h), vx=rng.uniform(-3, 3),
                 vy=rng.uniform(-0.6, 0.6), r=rng.uniform(2.5, 5.0), ph=rng.uniform(0, 6.3))
            for _ in range(scn.distractors)]
    rmask = np.zeros((h, w), np.uint8)
    if scn.occlusion:
        cv2.fillPoly(rmask, [rock], 255)
    for t in range(n_frames):
        s = t / n_frames
        # Lissajous path; the buoy grows as the ROV "approaches" it
        cx = w * (0.5 + 0.36 * np.sin(2 * np.pi * 1.3 * s + 0.4))
        cy = h * (0.48 + 0.26 * np.sin(2 * np.pi * 2.1 * s))
        r = 14 + 34 * (0.5 - 0.5 * np.cos(2 * np.pi * s))
        f = bg.copy()
        _sphere(f, cx, cy, r)
        for fi in fish:
            fi["x"] = (fi["x"] + fi["vx"]) % w
            fi["y"] += fi["vy"] + 0.6 * np.sin(t / 7 + fi["ph"])
            cv2.ellipse(f, (int(fi["x"]), int(fi["y"])), (int(fi["r"] * 1.8), int(fi["r"])), 0, 0, 360,
                        (30, 120, 235), -1, cv2.LINE_AA)
        for _ in range(scn.particles):
            px, py = rng.integers(0, w), rng.integers(0, h)
            f[py:py + rng.integers(1, 3), px:px + rng.integers(1, 3)] = (40, 90, 200)
        disk = np.zeros((h, w), np.uint8)
        cv2.circle(disk, (int(round(cx)), int(round(cy))), int(r), 255, -1)
        full = max(np.count_nonzero(disk), 1)
        if scn.occlusion:
            f[rmask > 0] = f[rmask > 0] * 0.25 + np.array([45, 55, 55], np.float32) * 0.75
            disk &= ~rmask
        visible = np.count_nonzero(disk) / full > 0.25   # mostly hidden behind the rock = not visible
        # water column effects
        f = f * (1 - scn.turbidity) + water * scn.turbidity
        f *= 1 + scn.lighting * np.sin(2 * np.pi * t / 37) * 0.8
        if scn.blur > 0:
            f = cv2.GaussianBlur(f, (0, 0), scn.blur)
        f += rng.normal(0, scn.noise, f.shape).astype(np.float32)
        gt = dict(cx=float(cx), cy=float(cy), r=float(r), visible=bool(visible), mask=disk)
        yield np.clip(f, 0, 255).astype(np.uint8), gt


def sample_frame(scenario: str = "clear", t: int = 40) -> np.ndarray:
    for i, (frame, _) in enumerate(generate(SCENARIOS[scenario], n_frames=240)):
        if i == t:
            return frame
    raise IndexError(t)
