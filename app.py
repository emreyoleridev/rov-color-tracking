"""Streamlit app: color-based object tracking for ROV navigation.

Run:  streamlit run app.py
"""
import re
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from tracker import (PRESETS, SCENARIOS, ColorTracker, TrackerConfig, compute_command, draw_overlay, generate,
                     process_image, sample_frame)
from tracker.video import read_frames, write_mp4

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
HSV_KEYS = ("h_lo", "s_lo", "v_lo", "h_hi", "s_hi", "v_hi")

st.set_page_config(page_title="ROV Color Tracking", page_icon="🎯", layout="wide")


def rgb(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode(data: bytes, max_w: int = 960) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    s = max_w / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def set_hsv(lo, hi):
    for k, v in zip(HSV_KEYS, (*lo, *hi)):
        st.session_state[k] = int(v)


def apply_preset():
    if st.session_state.preset in PRESETS:
        set_hsv(*PRESETS[st.session_state.preset])


def calibrate(img: np.ndarray, roi: tuple[int, int, int, int]):
    """Set the HSV range from the 5th-95th percentiles of a region; handles red hue wrap-around."""
    x, y, w, h = roi
    hsv = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    hue = hsv[:, 0]
    lo_h, hi_h = np.percentile(hue, [5, 95])
    shifted = (hue + 90) % 180
    lo_s, hi_s = np.percentile(shifted, [5, 95])
    if hi_s - lo_s < hi_h - lo_h:      # tighter when shifted -> range crosses 0 (red)
        lo_h, hi_h = (lo_s - 90) % 180, (hi_s - 90) % 180
    lo_h, hi_h = (lo_h - 4) % 180, (hi_h + 4) % 180
    s_lo, v_lo = np.maximum(np.percentile(hsv[:, 1:], 5, axis=0) - 25, 0)
    set_hsv((lo_h, s_lo, v_lo), (hi_h, 255, 255))
    st.session_state.preset = "Custom"


if "h_lo" not in st.session_state:
    st.session_state.preset = "Orange buoy"
    apply_preset()

# --- Sidebar: tracker parameters ---------------------------------------------------------------
st.sidebar.title("🎯 ROV Color Tracking")
st.sidebar.selectbox("Target color preset", [*PRESETS, "Custom"], key="preset", on_change=apply_preset,
                     help="Pick a preset, then fine-tune the HSV range below.")
with st.sidebar.expander("HSV range", expanded=True):
    st.caption("OpenCV units: H 0-179, S/V 0-255. If **H low > H high** the hue range wraps around 0 (red).")
    c1, c2 = st.columns(2)
    c1.slider("H low", 0, 179, key="h_lo")
    c2.slider("H high", 0, 179, key="h_hi")
    c1.slider("S low", 0, 255, key="s_lo")
    c2.slider("S high", 0, 255, key="s_hi")
    c1.slider("V low", 0, 255, key="v_lo")
    c2.slider("V high", 0, 255, key="v_hi")
with st.sidebar.expander("Morphology & contours", expanded=False):
    morph = st.checkbox("Morphological filtering", True)
    blur = st.slider("Gaussian blur kernel", 0, 15, 5, 2)
    open_k = st.slider("Opening kernel (removes specks)", 1, 21, 5, 2)
    close_k = st.slider("Closing kernel (fills holes)", 1, 31, 9, 2)
    iters = st.slider("Iterations", 1, 4, 1)
    min_area = st.slider("Min contour area (px)", 1, 3000, 150, 10)
with st.sidebar.expander("Guidance", expanded=False):
    deadzone = st.slider("Dead zone (fraction of half-frame)", 0.0, 0.5, 0.10, 0.01,
                         help="Offsets smaller than this count as centered (no yaw/heave correction).")
    stop_area = st.slider("Stop when target covers (% of frame)", 1.0, 40.0, 8.0, 0.5) / 100
    smoothing = st.slider("Smoothing α (1 = none)", 0.05, 1.0, 0.5, 0.05,
                          help="Exponential moving average weight of the newest measurement (video only).")
    max_lost = st.slider("Coast frames before SEARCH", 0, 60, 10)

ss = st.session_state
cfg = TrackerConfig(lower=(ss.h_lo, ss.s_lo, ss.v_lo), upper=(ss.h_hi, ss.s_hi, ss.v_hi), blur=blur,
                    open_kernel=open_k, close_kernel=close_k, iterations=iters, min_area=min_area,
                    deadzone=deadzone, stop_area=stop_area, smoothing=smoothing, max_lost=max_lost,
                    morphology=morph)

tab_img, tab_vid, tab_bench, tab_report = st.tabs(["🖼️ Image / Camera", "🎬 Video", "📊 Benchmark", "📄 Report"])


# --- Image tab ---------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_sample(scn: str, t: int) -> np.ndarray:
    return sample_frame(scn, t)


with tab_img:
    src = st.radio("Source", ["Synthetic underwater frame", "Upload image", "Camera snapshot"], horizontal=True)
    img = None
    if src == "Synthetic underwater frame":
        c1, c2 = st.columns(2)
        scn = c1.selectbox("Scenario", list(SCENARIOS), index=5)
        t = c2.slider("Frame", 0, 239, 70)
        img = cached_sample(scn, t)
    elif src == "Upload image":
        up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp", "webp"])
        img = decode(up.getvalue()) if up else None
    else:
        shot = st.camera_input("Hold a colored object in front of the camera")
        img = decode(shot.getvalue()) if shot else None

    if img is None:
        st.info("Choose an image to start.")
    else:
        res = process_image(img, cfg)
        cmd, det = res.command, res.detection
        m = st.columns(5)
        m[0].metric("State", cmd.state)
        m[1].metric("Command", cmd.text)
        m[2].metric("dx / dy", f"{cmd.dx:+.2f} / {cmd.dy:+.2f}" if det else "–")
        m[3].metric("Target area", f"{100 * det.area / (img.shape[0] * img.shape[1]):.2f} %" if det else "–")
        m[4].metric("Candidates", len(res.candidates))

        c1, c2 = st.columns(2)
        c1.image(rgb(draw_overlay(img, res, cfg)), caption="Tracking overlay", width="stretch")
        c2.image(rgb(cv2.bitwise_and(img, img, mask=res.mask)), caption="Segmented target (after morphology)",
                 width="stretch")
        c1.image(res.mask_raw, caption="Raw HSV threshold", width="stretch")
        c2.image(res.mask, caption="Cleaned mask (open → close)", width="stretch")
        if det:
            st.caption(f"Centroid ({det.cx:.0f}, {det.cy:.0f}) px · area {det.area:.0f} px · "
                       f"enclosing radius {det.radius:.0f} px · circularity {det.circularity:.2f}")

        with st.expander("🎨 Auto-calibrate HSV from a region of this image"):
            h, w = img.shape[:2]
            c1, c2, c3 = st.columns(3)
            rx = c1.slider("Region center x (%)", 0, 100, int(100 * det.cx / w) if det else 50)
            ry = c2.slider("Region center y (%)", 0, 100, int(100 * det.cy / h) if det else 50)
            rs = c3.slider("Region size (% of width)", 2, 50, 6)
            size = max(4, int(rs * w / 100))
            x0 = int(np.clip(rx * w / 100 - size / 2, 0, w - size))
            y0 = int(np.clip(ry * h / 100 - size / 2, 0, h - size))
            prev = img.copy()
            cv2.rectangle(prev, (x0, y0), (x0 + size, y0 + size), (255, 0, 255), 2)
            st.image(rgb(prev), width=480)
            st.button("Use this region's color", on_click=calibrate, args=(img, (x0, y0, size, size)),
                      type="primary")


# --- Video tab ---------------------------------------------------------------------------------
def run_video(frames, fps: float, n_total: int, gts=None):
    tr = ColorTracker(cfg)
    rows, out_frames = [], []
    bar = st.progress(0.0, "Tracking…")
    elapsed = 0.0
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        res = tr.update(frame)
        elapsed += time.perf_counter() - t0
        c = res.command
        row = dict(frame=i, time_s=i / fps, state=c.state, command=c.text, yaw=c.yaw, heave=c.heave,
                   surge=c.surge, dx=c.dx, dy=c.dy, yaw_rate=c.yaw_rate, heave_rate=c.heave_rate,
                   cx=res.estimate[0] if res.estimate else np.nan, cy=res.estimate[1] if res.estimate else np.nan)
        if gts is not None:
            gt = gts[i]
            g = compute_command(gt["cx"], gt["cy"], np.pi * gt["r"] ** 2, frame.shape, cfg)
            row |= dict(gt_visible=gt["visible"], gt_dx=g.dx, gt_dy=g.dy, gt_command=g.text,
                        err_px=np.hypot(row["cx"] - gt["cx"], row["cy"] - gt["cy"]))
        rows.append(row)
        over = draw_overlay(frame, res, cfg)
        if ss.get("side_mask"):
            over = np.hstack([over, cv2.cvtColor(res.mask, cv2.COLOR_GRAY2BGR)])
        out_frames.append(over)
        bar.progress(min((i + 1) / max(n_total, 1), 1.0), f"Tracking… frame {i + 1}/{n_total}")
    bar.progress(1.0, "Encoding video…")
    path = Path(tempfile.mkdtemp()) / "tracked.mp4"
    write_mp4(str(path), out_frames, fps)
    bar.empty()
    return dict(video=path.read_bytes(), log=pd.DataFrame(rows), fps=len(rows) / max(elapsed, 1e-9),
                synthetic=gts is not None)


with tab_vid:
    vsrc = st.radio("Source", ["Synthetic underwater scenario", "Upload video"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    max_frames = c3.slider("Max frames", 30, 900, 240, 30)
    st.checkbox("Show mask next to the video", key="side_mask")
    go = False
    if vsrc == "Synthetic underwater scenario":
        scn = c1.selectbox("Scenario ", list(SCENARIOS), index=5,
                           help="clear / turbid / noisy / distractors / occlusion / hard (all effects combined)")
        seed = c2.number_input("Random seed", 0, 999, 0)
        go = st.button("▶ Generate & track", type="primary")
        if go:
            with st.spinner("Rendering synthetic video…"):
                data = list(generate(SCENARIOS[scn], max_frames, seed=int(seed)))
            ss.video_result = run_video((f for f, _ in data), 30.0, len(data), [g for _, g in data])
    else:
        up = c1.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv", "webm"])
        go = st.button("▶ Track uploaded video", type="primary", disabled=up is None)
        if go and up:
            tmp = Path(tempfile.mkdtemp()) / up.name
            tmp.write_bytes(up.getvalue())
            try:
                frames, fps, n = read_frames(str(tmp), max_frames=max_frames)
                ss.video_result = run_video(frames, fps, n)
            except ValueError as e:
                st.error(str(e))

    r = ss.get("video_result")
    if r:
        log = r["log"]
        m = st.columns(5)
        m[0].metric("Frames", len(log))
        m[1].metric("Tracking", f"{100 * (log.state == 'TRACKING').mean():.0f} %")
        m[2].metric("Searching", f"{100 * (log.state == 'SEARCHING').mean():.0f} %")
        m[3].metric("Tracker speed", f"{r['fps']:.0f} FPS")
        if r["synthetic"]:
            vis = log[log.gt_visible]
            m[4].metric("Command accuracy vs GT", f"{100 * (vis.command == vis.gt_command).mean():.1f} %",
                        help="Share of frames (target visible) where yaw+heave+surge match the ground truth.")
        else:
            m[4].metric("Command changes", int((log.command != log.command.shift()).sum() - 1))
        c1, c2 = st.columns([3, 2])
        c1.video(r["video"])
        c2.markdown("**Normalized offset from image center**")
        cols = ["dx", "dy"] + (["gt_dx", "gt_dy"] if r["synthetic"] else [])
        c2.line_chart(log.set_index("frame")[cols], height=200)
        c2.markdown("**Command distribution**")
        c2.bar_chart(log.command.value_counts(), height=200, horizontal=True)
        if r["synthetic"]:
            c2.caption(f"Mean centroid error while visible: {log[log.gt_visible].err_px.mean():.1f} px")
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Annotated video (MP4)", r["video"], "tracked.mp4", "video/mp4")
        d2.download_button("⬇ Command log (CSV)", log.to_csv(index=False), "commands.csv", "text/csv")
        with st.expander("Command log"):
            st.dataframe(log, width="stretch", height=300)


# --- Benchmark tab -----------------------------------------------------------------------------
with tab_bench:
    if (RESULTS / "summary.csv").exists():
        st.markdown("Results of `python scripts/evaluate.py` on synthetic scenarios with exact ground truth.")
        st.dataframe(pd.read_csv(RESULTS / "summary.csv"), width="stretch", hide_index=True)
        for name, cap in [("scenarios.jpg", "Scenarios"), ("pipeline_stages.jpg", "Pipeline stages"),
                          ("trajectory_error.png", "Trajectory and error"), ("dx_command.png", "Yaw offset"),
                          ("ablation.png", "Ablation"), ("sensitivity_smin.png", "Saturation sensitivity")]:
            if (RESULTS / "figures" / name).exists():
                st.image(str(RESULTS / "figures" / name), caption=cap, width="stretch")
    else:
        st.info("Run `python scripts/evaluate.py` to produce benchmark results.")


# --- Report tab --------------------------------------------------------------------------------
with tab_report:
    report = ROOT / "REPORT.md"
    if report.exists():
        img_re = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
        buf: list[str] = []
        for line in report.read_text().splitlines():
            m_ = img_re.match(line)
            if m_ and (ROOT / m_.group(2)).exists():
                st.markdown("\n".join(buf)); buf = []
                st.image(str(ROOT / m_.group(2)), caption=m_.group(1), width="stretch")
            else:
                buf.append(line)
        st.markdown("\n".join(buf))
