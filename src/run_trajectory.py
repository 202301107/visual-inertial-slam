"""Marker-free trajectory for one run (the M4 baseline): VO + IMU, no marker correction.

    python -m src.run_trajectory "data/all runs/run 1"

Writes outputs/<run>_traj.csv and outputs/<run>_traj.png (spaces in the run name become '_').
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_markers import DEFAULT_CALIB_PATH, load_calibration
from src.odometry.trajectory import Trajectory, TrajectoryConfig, build_trajectory
from src.odometry.visual_odometry import VisualOdometry


def run_vo(ds: RunDataset, K, dist, scale: float, step: int = 200, progress=None):
    vo = VisualOdometry(K, dist, scale=scale)
    steps = []
    t0 = time.time()
    for idx, _, frame in ds.frames():
        s = vo.step(frame, idx)
        if s is not None:
            steps.append(s)
        if progress is not None:
            progress(idx, len(ds.frame_t_ns))
        if idx % step == 0:
            print(f"  VO frame {idx}/{len(ds.frame_t_ns)}  ({time.time() - t0:.0f}s)", file=sys.stderr)
    return steps


def plot_topdown(tr: Trajectory, title: str, path: Path) -> None:
    t = (tr.t_ns - tr.t_ns[0]) / 1e9
    xy = tr.p[:, :2]
    fig, ax = plt.subplots(figsize=(8, 8))
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", linewidth=2.2)
    lc.set_array(0.5 * (t[:-1] + t[1:]))
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, shrink=0.75, label="time (s)")

    stride = max(len(t) // 25, 1)                    # camera heading, horizontal projection
    fwd = tr.R_wc[:, :, 2][:, :2]
    nrm = np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    idx = np.arange(0, len(t), stride)
    ax.quiver(xy[idx, 0], xy[idx, 1], (fwd / nrm)[idx, 0], (fwd / nrm)[idx, 1], color="0.35",
              width=0.003, scale=30, alpha=0.7, label="camera heading")
    ax.plot(*xy[0], "o", color="tab:green", ms=11, label="start")
    ax.plot(*xy[-1], "s", color="tab:red", ms=10, label="end")

    unit = "m" if tr.metric else "arbitrary units"
    ax.set_xlabel(f"X ({unit})  [heading of camera at start]")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    scale_note = ("metric scale from IMU integration (unreliable, expect large scale error)"
                  if tr.metric else "ARBITRARY scale: 1 unit per VO step")
    ax.set_title(f"{title}\nno marker correction | {scale_note}\n"
                 f"path length {tr.path_length:.2f} {unit}, net displacement "
                 f"{np.linalg.norm(xy[-1] - xy[0]):.2f} {unit}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--mode", choices=["vio", "vo"], default="vio",
                    help="vio: gyro orientation + VO direction + IMU speed (default)\n"
                         "vo: VO-only, arbitrary scale")
    ap.add_argument("--vo-scale", type=float, default=1.0, help="downscale frames for ORB (e.g. 0.5 is ~3x faster)")
    ap.add_argument("--no-gyro-bias", action="store_true", help="do not subtract the static-start gyro bias")
    ap.add_argument("--vel-leak", type=float, default=TrajectoryConfig.vel_leak_tau_s,
                    help="velocity decay time constant in s (vio mode)")
    a = ap.parse_args(argv)

    ds = RunDataset(a.run_dir)
    print(ds.summary())
    K, dist, _ = load_calibration(a.calib)

    steps = run_vo(ds, K, dist, a.vo_scale)
    ok = np.mean([s.ok for s in steps])
    print(f"VO: {len(steps)} steps, {100 * ok:.0f}% estimated")

    cfg = TrajectoryConfig(mode=a.mode, gyro_bias="none" if a.no_gyro_bias else "static",
                           vel_leak_tau_s=a.vel_leak)
    tr = build_trajectory(ds, steps, K, cfg)

    name = Path(a.run_dir).resolve().name.strip().replace(" ", "_")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, png_path = out / f"{name}_traj.csv", out / f"{name}_traj.png"
    tr.save_csv(csv_path)
    plot_topdown(tr, f"{name} - top-down trajectory ({a.mode})", png_path)

    unit = "m" if tr.metric else "units"
    print(f"\nframes: {len(tr.t_ns)}  path length: {tr.path_length:.2f} {unit}  "
          f"net displacement: {np.linalg.norm(tr.p[-1] - tr.p[0]):.2f} {unit}")
    print(f"frames with a reliable VO direction: {100 * tr.dir_ok.mean():.0f}%  "
          f"stationary (ZUPT): {100 * tr.stationary.mean():.0f}%")
    for k in ("gyro_bias_rad_s", "gravity_norm", "R_ci", "extrinsic_score_margin"):
        if k in tr.info:
            print(f"{k}: {tr.info[k]}")
    print(f"\nsaved {csv_path}\nsaved {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
