#!/usr/bin/env python3
"""Animate the progress chart for exp 013, adding one run per frame."""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


RESULTS = Path(__file__).parent / "results_1x.tsv"
OUTPUT_MP4 = Path(__file__).parent / "progress_1x_animated.mp4"
OUTPUT_GIF = Path(__file__).parent / "progress_1x_animated.gif"

KEPT_STATUSES = {"KEEP", "KEPT"}
SKIP_STATUSES = {"CRASH", "ERROR", "FAIL", "FAILED", "TIMEOUT", "OOM"}

WIDTH, HEIGHT = 16, 9
DPI = 200
FPS = 3
HOLD_LAST_FRAMES = 8


def parse_float(v):
    if not v or not v.strip():
        return None
    try:
        n = float(v.strip().replace(",", ""))
        return n if math.isfinite(n) else None
    except ValueError:
        return None


def load_data():
    rows = []
    with open(RESULTS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            status = (row.get("status") or "").strip().upper()
            metric = parse_float(row.get("val_bpb"))
            desc = " ".join((row.get("description") or "").split())
            variant = (row.get("variant") or "").strip()
            if status in SKIP_STATUSES or metric is None:
                continue
            rows.append({
                "x": i,
                "metric": metric,
                "status": status,
                "kept": status in KEPT_STATUSES,
                "desc": desc[:50],
                "variant": variant,
            })
    return rows


def running_best(points):
    xs, ys = [], []
    best = None
    for p in points:
        if best is None or p["metric"] < best:
            best = p["metric"]
        xs.append(p["x"])
        ys.append(best)
    return xs, ys


def compute_ylim(all_rows):
    metrics = [r["metric"] for r in all_rows]
    lo = min(metrics)
    hi = max(metrics)
    margin = (hi - lo) * 0.12
    return lo - margin, hi + margin


def main():
    rows = load_data()
    if not rows:
        print("No plottable data.")
        return

    y_lo, y_hi = compute_ylim(rows)
    x_max = max(r["x"] for r in rows)

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    def draw_frame(n):
        ax.clear()
        visible = rows[:n + 1]

        kept_pts = [p for p in visible if p["kept"]]
        disc_pts = [p for p in visible if not p["kept"]]

        if disc_pts:
            ax.scatter(
                [p["x"] for p in disc_pts],
                [p["metric"] for p in disc_pts],
                c="#cccccc", s=24, alpha=0.5, zorder=2, label="Discarded",
            )
        if kept_pts:
            ax.scatter(
                [p["x"] for p in kept_pts],
                [p["metric"] for p in kept_pts],
                c="#2ecc71", s=80, zorder=4, label="Kept",
                edgecolors="black", linewidths=0.5,
            )
            bx, by = running_best(kept_pts)
            ax.step(bx, by, where="post", color="#27ae60", linewidth=2.5, alpha=0.7, zorder=3, label="Running best")

            for p in kept_pts:
                ax.annotate(
                    p["desc"][:40],
                    (p["x"], p["metric"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color="#1a7a3a", alpha=0.85,
                    rotation=25, ha="left", va="bottom",
                )

        current = visible[-1]
        ax.scatter([current["x"]], [current["metric"]],
                   c="#e74c3c" if not current["kept"] else "#2ecc71",
                   s=180, zorder=5, edgecolors="black", linewidths=2, marker="D")

        n_total = len(visible)
        n_kept = len(kept_pts)
        best_val = min((p["metric"] for p in kept_pts), default=None) if kept_pts else None
        title = f"Exp 013 AR-Latent Diffusion — Run {n_total}/{len(rows)}"
        if best_val is not None:
            title += f"  |  Best: {best_val:.4f}"
        title += f"  |  Kept: {n_kept}"

        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("Run #", fontsize=14)
        ax.set_ylabel("val_bpb (lower is better)", fontsize=14)
        ax.set_xlim(-1, x_max + 2)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=11)
        ax.tick_params(labelsize=12)

    n_frames = len(rows) + HOLD_LAST_FRAMES

    def animate(frame_idx):
        data_idx = min(frame_idx, len(rows) - 1)
        draw_frame(data_idx)

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000 // FPS, repeat=False)

    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Saving MP4 to {OUTPUT_MP4} ...")
        writer = animation.FFMpegWriter(fps=FPS, bitrate=5000)
        anim.save(str(OUTPUT_MP4), writer=writer, dpi=DPI)
        print(f"Done: {OUTPUT_MP4} ({OUTPUT_MP4.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"MP4 failed ({e}), falling back to GIF...")

    try:
        print(f"Saving GIF to {OUTPUT_GIF} ...")
        writer = animation.PillowWriter(fps=FPS)
        anim.save(str(OUTPUT_GIF), writer=writer, dpi=DPI)
        print(f"Done: {OUTPUT_GIF} ({OUTPUT_GIF.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"GIF failed ({e})")

    plt.close(fig)


if __name__ == "__main__":
    main()
