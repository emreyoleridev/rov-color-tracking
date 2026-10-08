"""Benchmark the tracker on synthetic underwater scenarios with exact ground truth.

    python scripts/evaluate.py            # writes results/*.csv, results/summary.md, results/figures/*
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracker import SCENARIOS, ColorTracker, TrackerConfig, compute_command, draw_overlay, generate  # noqa: E402
from tracker.video import write_mp4  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"
N_FRAMES, SEEDS = 300, (0, 1)

VARIANTS = {
    "Full pipeline": TrackerConfig(),
    "No morphology": TrackerConfig(morphology=False),
    "No smoothing": TrackerConfig(smoothing=1.0),
    "No min-area filter": TrackerConfig(min_area=1),
}


def run(scn_name: str, cfg: TrackerConfig, seed: int, keep: bool = False):
    tr = ColorTracker(cfg)
    rows, frames = [], []
    elapsed = 0.0
    for t, (frame, gt) in enumerate(generate(SCENARIOS[scn_name], N_FRAMES, seed=seed)):
        t0 = time.perf_counter()
        res = tr.update(frame)
        elapsed += time.perf_counter() - t0
        inter = np.count_nonzero(res.mask & gt["mask"])
        union = np.count_nonzero(res.mask | gt["mask"])
        d, e = res.detection, res.estimate
        gt_cmd = compute_command(gt["cx"], gt["cy"], np.pi * gt["r"] ** 2, frame.shape, cfg)
        det_err = np.hypot(d.cx - gt["cx"], d.cy - gt["cy"]) if d else np.nan
        est_err = np.hypot(e[0] - gt["cx"], e[1] - gt["cy"]) if e else np.nan
        c = res.command
        rows.append(dict(
            frame=t, visible=gt["visible"], iou=inter / union if union else np.nan,
            blobs=max(len(cv2.findContours(res.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) - 1, 0), gt_cx=gt["cx"], gt_cy=gt["cy"], gt_r=gt["r"],
            det=d is not None, det_cx=d.cx if d else np.nan, det_cy=d.cy if d else np.nan,
            est_cx=e[0] if e else np.nan, est_cy=e[1] if e else np.nan,
            det_err=det_err, est_err=est_err,
            hit=d is not None and det_err <= gt["r"],
            state=c.state, cmd=c.text, gt_cmd=gt_cmd.text, dx=c.dx, gt_dx=gt_cmd.dx,
            yaw_ok=c.yaw == gt_cmd.yaw, heave_ok=c.heave == gt_cmd.heave, surge_ok=c.surge == gt_cmd.surge))
        if keep:
            frames.append((frame, res))
    fps = N_FRAMES / elapsed
    return pd.DataFrame(rows), fps, frames


def score(df: pd.DataFrame) -> dict:
    vis, hid = df[df.visible], df[~df.visible]
    return {
        "Detection rate %": 100 * vis.hit.mean(),
        "False-positive frames %": 100 * ((df.det & ~df.hit) | (~df.visible & df.det)).mean(),
        "Mean centroid err (px)": vis.loc[vis.hit, "det_err"].mean(),
        "Smoothed err (px)": vis.loc[vis.hit, "est_err"].mean(),
        "Yaw cmd acc %": 100 * vis.yaw_ok.mean(),
        "Heave cmd acc %": 100 * vis.heave_ok.mean(),
        "Surge cmd acc %": 100 * vis.surge_ok.mean(),
        "Full cmd acc %": 100 * (vis.yaw_ok & vis.heave_ok & vis.surge_ok).mean(),
        "Mask IoU": vis.iou.mean(),
        "Spurious blobs/frame": df.blobs.mean(),
        "Cmd changes /100 fr": 100 * (df.cmd != df.cmd.shift()).iloc[1:].mean(),
        "Occluded frames": int(len(hid)),
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summary, ablation, logs = [], [], {}
    for scn in SCENARIOS:
        for vname, cfg in VARIANTS.items():
            scores, fpss = [], []
            for seed in SEEDS:
                df, fps, _ = run(scn, cfg, seed)
                scores.append(score(df)); fpss.append(fps)
                if vname == "Full pipeline" and seed == 0:
                    logs[scn] = df
            row = {"Scenario": scn, "Variant": vname, **pd.DataFrame(scores).mean().to_dict(), "FPS": np.mean(fpss)}
            ablation.append(row)
            if vname == "Full pipeline":
                summary.append(row)
        print(f"{scn:12s} done")

    s = pd.DataFrame(summary).drop(columns="Variant").round(2)
    a = pd.DataFrame(ablation).round(2)
    s.to_csv(OUT / "summary.csv", index=False)
    a.to_csv(OUT / "ablation.csv", index=False)
    pd.concat([d.assign(scenario=k) for k, d in logs.items()]).to_csv(OUT / "per_frame_log.csv", index=False)

    abl_cols = ["Detection rate %", "False-positive frames %", "Mean centroid err (px)", "Full cmd acc %",
                "Mask IoU", "Spurious blobs/frame", "Cmd changes /100 fr"]
    with open(OUT / "summary.md", "w") as f:
        f.write(f"# Results ({N_FRAMES} frames x {len(SEEDS)} seeds per scenario, 640x480)\n\n")
        f.write(s.to_markdown(index=False) + "\n\n## Ablation\n\n")
        for metric in abl_cols[1:]:
            f.write(f"### {metric}\n\n" + a.pivot(index="Scenario", columns="Variant", values=metric)
                    .loc[list(SCENARIOS), list(VARIANTS)].round(2).to_markdown() + "\n\n")
    print(s.to_string(index=False))
    print(a[["Scenario", "Variant", *abl_cols]].to_string(index=False))

    sens = sensitivity()
    sens.to_csv(OUT / "sensitivity_smin.csv", index=False)
    with open(OUT / "summary.md", "a") as f:
        f.write("\n## Sensitivity to the saturation threshold (S_min)\n\n" + sens.round(2).to_markdown(index=False) + "\n")
    print(sens.round(2).to_string(index=False))

    figures(logs, a, sens)


def sensitivity(scenarios=("clear", "turbid", "hard"), smins=(30, 50, 80, 110, 140, 170)) -> pd.DataFrame:
    rows = []
    base = TrackerConfig()
    for scn in scenarios:
        for smin in smins:
            cfg = replace(base, lower=(base.lower[0], smin, base.lower[2]))
            sc = score(run(scn, cfg, 0)[0])
            rows.append({"Scenario": scn, "S_min": smin, **{k: sc[k] for k in (
                "Mask IoU", "Mean centroid err (px)", "Spurious blobs/frame", "False-positive frames %")}})
    return pd.DataFrame(rows)


def figures(logs: dict, ablation: pd.DataFrame, sens: pd.DataFrame):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    cfg = TrackerConfig()

    # 1. pipeline stages on the hardest scenario
    _, _, frames = run("hard", cfg, 0, keep=True)
    t = 75
    frame, res = frames[t]
    stages = [("Input frame", frame), ("HSV threshold", res.mask_raw), ("After open + close", res.mask),
              ("Contour + command", draw_overlay(frame, res, cfg))]
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.9))
    for ax, (title, img) in zip(axes, stages):
        ax.imshow(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap="gray")
        ax.set_title(title, loc="left"); ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "pipeline_stages.jpg", dpi=130); plt.close(fig)

    # demo video of the hard scenario
    write_mp4(str(OUT / "demo_hard.mp4"), (draw_overlay(f, r, cfg) for f, r in frames), fps=30)
    cv2.imwrite(str(FIG / "overlay_example.jpg"), draw_overlay(frame, res, cfg))

    # 2. trajectory + error over time (hard)
    df = logs["hard"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    a1.plot(df.gt_cx, df.gt_cy, color="#9aa0a6", lw=3, label="Ground truth")
    a1.plot(df.est_cx, df.est_cy, color="#e8710a", lw=1.2, label="Tracker (smoothed)")
    a1.set_xlim(0, 640); a1.set_ylim(480, 0); a1.set_aspect("equal"); a1.legend(frameon=False)
    a1.set_title("Target trajectory in image (hard scenario)", loc="left")
    a2.plot(df.frame, df.det_err.where(df.hit), color="#1a73e8", lw=1, label="Raw detection")
    a2.plot(df.frame, df.est_err.where(df.hit), color="#e8710a", lw=1, label="Smoothed")
    for start, end in _spans(~df.visible):
        a2.axvspan(start, end, color="#d0d0d0", alpha=0.6, lw=0)
    a2.set_xlabel("frame"); a2.set_ylabel("centroid error (px)")
    a2.set_title("Centroid error  (gray = target occluded)", loc="left"); a2.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "trajectory_error.png", dpi=130); plt.close(fig)

    # 3. horizontal offset command vs ground truth
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.plot(df.frame, df.gt_dx, color="#9aa0a6", lw=3, label="Ground-truth dx")
    ax.plot(df.frame, df.dx, color="#e8710a", lw=1.2, label="Tracker dx")
    ax.axhspan(-cfg.deadzone, cfg.deadzone, color="#1a73e8", alpha=0.08, lw=0, label="Dead zone (CENTER)")
    ax.set_xlabel("frame"); ax.set_ylabel("dx"); ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title("Normalized horizontal offset driving the yaw command (hard scenario)", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "dx_command.png", dpi=130); plt.close(fig)

    # 4. ablation
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
    colors = {"Full pipeline": "#e8710a", "No morphology": "#1a73e8", "No smoothing": "#12b5cb",
              "No min-area filter": "#9334e6"}
    for ax, metric in zip(axes, ["Spurious blobs/frame", "False-positive frames %",
                                                   "Cmd changes /100 fr"]):
        p = ablation.pivot(index="Scenario", columns="Variant", values=metric).loc[list(SCENARIOS)]
        x = np.arange(len(p))
        for i, v in enumerate(colors):
            ax.bar(x + (i - 1.5) * 0.2, p[v], 0.2, label=v, color=colors[v])
        ax.set_xticks(x, p.index); ax.set_title(metric, loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "ablation.png", dpi=130); plt.close(fig)

    # 5. saturation-threshold sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for scn, col in zip(sens.Scenario.unique(), ["#1a73e8", "#e8710a", "#9334e6"]):
        d = sens[sens.Scenario == scn]
        axes[0].plot(d.S_min, d["Mask IoU"], "o-", color=col, label=scn)
        axes[1].plot(d.S_min, d["Mean centroid err (px)"], "o-", color=col, label=scn)
    for ax, t in zip(axes, ["Mask IoU vs S_min", "Centroid error (px) vs S_min"]):
        ax.axvline(TrackerConfig().lower[1], color="#9aa0a6", ls="--", lw=1)
        ax.set_xlabel("S_min (HSV lower saturation)"); ax.set_title(t, loc="left")
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "sensitivity_smin.png", dpi=130); plt.close(fig)

    # 6. one frame per scenario
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, scn in zip(axes.ravel(), SCENARIOS):
        tr = ColorTracker(cfg)
        for i, (f, _) in enumerate(generate(SCENARIOS[scn], N_FRAMES, seed=0)):
            r = tr.update(f)
            if i == 150:
                break
        ax.imshow(cv2.cvtColor(draw_overlay(f, r, cfg), cv2.COLOR_BGR2RGB)); ax.set_title(scn, loc="left")
        ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "scenarios.jpg", dpi=110); plt.close(fig)


def _spans(mask: pd.Series):
    v = mask.to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, v.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


if __name__ == "__main__":
    main()
