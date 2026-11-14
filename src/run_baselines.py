"""Five-configuration baseline comparison on one run (default usage: run 1), algorithm = only variable.

    python -m src.run_baselines "data/all runs/run 1" --map outputs/marker_map_excl_run1.json \\
        --gt-path "2.9,3.25;0.5,3.25;0.5,0.25"

Configurations (same video, same IMU, same cached detections, same keyframes):
  1 IMU only                      strap-down gyro + accelerometer dead reckoning
  2 Visual odometry only          chained essential-matrix poses, unit steps (arbitrary scale)
  3 Visual-inertial, no markers   the M4 trajectory
  4 Visual-inertial + markers     pose graph with the (leave-this-run-out) marker map
  5 ... + loop closure            same plus explicit relative-pose edges between re-observations of a marker

Registration: every configuration starts from the same marker-derived pose (yaw + translation fitted over
the first second of accepted marker sightings, ~8 s into the run), then is left alone. Configs 4 and 5
keep using markers afterwards. Config 2 is scaled so its path length equals the taped path length (an
oracle favouring VO, since monocular scale is unobservable).
Metrics: endpoint drift = distance from the estimated final position to the taped end point (this includes
a constant ~0.3-0.5 m offset because the phone is held ahead of the body that walks the tape, identical for
all configs); drift per metre = endpoint drift / taped path length; loop residual = mean gap between two
estimates of the same marker's position made on different visits (>= --loop-min-gap s apart) from each
config's own camera poses.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_markers import load_calibration
from src.detection.detect_run import detect_run
from src.estimation.loop_closure import split_visits
from src.estimation.pose_graph import MarkerObs
from src.estimation.slam_core import solve_with_map
from src.geometry import inv_T, make_T, T_from_rvec_tvec
from src.mapping.marker_map import MapConfig, MarkerMap, _gate
from src.odometry.trajectory import TrajectoryConfig, build_trajectory
from src.run_slam import parse_polyline, polyline_error
from src.run_trajectory import run_vo

NAMES = ["IMU only\n(dead reckoning)", "Visual odometry\nonly", "Visual-inertial,\nno markers",
         "Visual-inertial\n+ markers", "Visual-inertial + markers\n+ loop closure"]


def yaw_of(R):
    return np.arctan2(R[..., 1, 2], R[..., 0, 2])           # heading of the camera optical axis


def anchor_poses(det, markers, use_ids, window_frames=30):
    """Marker-derived camera poses (room frame) for the first `window_frames` frames of the first sighting."""
    keep = np.isin(det.marker_id, list(use_ids))
    f0 = int(det.frame[keep].min())
    poses = {}
    for f, m, rv, tv, d in zip(det.frame, det.marker_id, det.rvec, det.tvec, det.distance):
        if m in use_ids and f0 <= f < f0 + window_frames and (f not in poses or d < poses[f][1]):
            poses[int(f)] = (markers[int(m)] @ inv_T(T_from_rvec_tvec(rv, tv)), d)
    return f0, {f: v[0] for f, v in poses.items()}


def register(R, p, anchor):
    """Yaw + translation that best put trajectory poses (R, p arrays indexed by frame) on the anchor poses."""
    fr = sorted(anchor)
    dpsi = np.array([yaw_of(anchor[f][:3, :3]) - yaw_of(R[f]) for f in fr])
    psi = float(np.arctan2(np.sin(dpsi).mean(), np.cos(dpsi).mean()))
    Rz = Rotation.from_euler("z", psi).as_matrix()
    t = np.mean([anchor[f][:3, 3] - Rz @ p[f] for f in fr], axis=0)
    return np.einsum("ij,njk->nik", Rz, R), (Rz @ p.T).T + t


def loop_residual(frames, cam_R, cam_p, det_all, kf_t_ns, min_gap_s):
    """Mean gap between a marker's position as estimated on two visits >= min_gap_s apart (m), and #pairs."""
    fidx = {int(f): i for i, f in enumerate(frames)}
    obs, pos = [], {}
    for f, m, tv, d in zip(det_all.frame, det_all.marker_id, det_all.tvec, det_all.distance):
        k = fidx.get(int(f))
        if k is not None:
            o = MarkerObs(k, int(m), np.eye(3), tv, float(d))
            obs.append(o)
            pos[(k, int(m))] = cam_p[k] + cam_R[k] @ tv
    gaps = []
    for m, visits in split_visits(obs, kf_t_ns, 2.0).items():
        est = [np.median([pos[(o.kf, m)] for o in v], axis=0) for v in visits]
        for a in range(len(visits) - 1):
            if (kf_t_ns[visits[a + 1][0].kf] - kf_t_ns[visits[a][-1].kf]) * 1e-9 >= min_gap_s:
                gaps.append(float(np.linalg.norm(est[a + 1] - est[a])))
    return (float(np.mean(gaps)) if gaps else float("nan")), len(gaps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--map", required=True, help="marker map that does NOT include this run (e.g. outputs/marker_map_excl_run1.json)")
    ap.add_argument("--gt-path", required=True, help='taped ground-truth path "x,y;x,y;..." in room metres (room.md)')
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--loop-min-gap", type=float, default=3.0, help="s between visits that count as a revisit (run 1 has no real circuit)")
    ap.add_argument("--force-vo", action="store_true", help="recompute the cached visual odometry")
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    name = Path(a.run_dir).resolve().name.strip().replace(" ", "_")
    gt_poly = parse_polyline(a.gt_path)
    gt_len = float(np.linalg.norm(np.diff(gt_poly, axis=0), axis=1).sum())
    gt_end = gt_poly[-1]

    ds = RunDataset(a.run_dir)
    K, dist, _ = load_calibration()
    vo_cache = out / f"{name}_vo.pkl"
    if vo_cache.is_file() and not a.force_vo:
        steps = pickle.load(open(vo_cache, "rb"))
    else:
        steps = run_vo(ds, K, dist, 1.0)
        pickle.dump(steps, open(vo_cache, "wb"))
    print(f"{name}: {len(steps)} VO steps; taped path {gt_len:.2f} m, end {gt_end}")

    cfg = MapConfig()
    mmap = MarkerMap.load(a.map)
    markers = {m: mmap.T_room_world @ T for m, T in mmap.poses.items()}
    use_ids = {m for m, f in mmap.in_map.items() if f}
    det_all = detect_run(ds, cache_path=out / f"{name}_detections.npz")
    det = _gate(det_all, cfg)

    trajs = {i: build_trajectory(ds, steps, K, TrajectoryConfig(mode=mode)) for i, mode in ((0, "imu"), (1, "vo"), (2, "vio"))}
    f0, anchor = anchor_poses(det, markers, use_ids)
    print(f"registration: first accepted map-marker sighting at frame {f0} (t={f0 / 30:.1f} s), {len(anchor)} frames used")

    sol4 = solve_with_map(trajs[2], det, markers, use_ids, cfg, loop_closure=False)
    sol5 = solve_with_map(trajs[2], det, markers, use_ids, cfg, loop_closure=True, loop_min_gap_s=a.loop_min_gap)
    print(f"loop-closure edges found for config 5: {len(sol5.loops)} "
          f"({sorted({c.marker for c in sol5.loops})}); outlier observations dropped: {sol4.dropped_obs}/{sol5.dropped_obs}")
    frames = sol4.gi.kf_frames
    kf_t = sol4.gi.kf_t_ns

    cams = {}
    for i in (0, 1, 2):
        tr = trajs[i]
        p = tr.p.copy()
        if i == 1:
            p *= gt_len / float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())      # oracle scale for monocular VO
        R, p = register(tr.R_wc, p, anchor)
        cams[i] = (R[frames], p[frames])
    cams[3] = (sol4.cam_R, sol4.cam_p)
    cams[4] = (sol5.cam_R, sol5.cam_p)

    rows = []
    for i in range(5):
        R, p = cams[i]
        drift = float(np.linalg.norm(p[-1, :2] - gt_end))
        lr, npairs = loop_residual(frames, R, p, det_all, kf_t, a.loop_min_gap)
        rows.append({"name": NAMES[i].replace("\n", " "), "drift": drift, "per_m": drift / gt_len, "loop": lr,
                     "path_err": float(polyline_error(p[:, :2], gt_poly).mean()), "pairs": npairs,
                     "length": float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())})
    print(f"loop residual uses {rows[0]['pairs']} revisit pair(s) (>= {a.loop_min_gap:g} s apart)")

    md = ["| configuration | endpoint drift (m) | drift per metre | loop residual (m) | mean distance to taped path (m) | estimated path length (m) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['drift']:.2f} | {r['per_m']:.2f} | {r['loop']:.2f} | {r['path_err']:.2f} | {r['length']:.1f} |")
    notes = [
        "",
        f"All five configurations: {name}, taped path {gt_len:.1f} m, end point {tuple(gt_end)}; map `{Path(a.map).name}` (built without this run).",
        "Endpoint drift includes a constant ~0.3-0.5 m offset (phone held ahead of the body walking the tape), the same for every row.",
        f"Every row is registered to the room frame from the same marker-derived pose at t = {f0 / 30:.1f} s; rows 4-5 keep using markers afterwards.",
        "Visual odometry only is rescaled to the taped path length (oracle scale; monocular scale is unobservable).",
        f"Loop residual = gap between two estimates of one marker's position from visits >= {a.loop_min_gap:g} s apart; "
        f"only {rows[0]['pairs']} revisit pair(s) exist in this run, so it is a weak statistic here (run 3 is the real loop test).",
    ]
    (out / "baseline_table.md").write_text("\n".join(md + notes) + "\n")
    print("\n" + "\n".join(md))

    ink, sub, grid, bar = "#1f2933", "#52606d", "#e4e7eb", "#2f6fb0"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, key, title, unit in ((axes[0], "drift", "Endpoint drift", "m"),
                                 (axes[1], "per_m", "Drift per metre walked", "m per m"),
                                 (axes[2], "loop", "Loop residual", "m")):
        vals = np.array([r[key] for r in rows])
        x = np.arange(5)
        ax.bar(x, vals, width=0.55, color=bar)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["IMU", "VO", "VIO", "VIO +\nmarkers", "VIO + markers\n+ loop cl."], fontsize=8, color=sub)
        for xi, v in zip(x, vals):
            ax.text(xi, v * 1.12, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=ink)
        ax.set_title(f"{title} ({unit}, log scale)", fontsize=11, color=ink, loc="left")
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", colors=sub, length=0)
    fig.suptitle(f"{name}: baseline configurations (lower is better; taped path {gt_len:.1f} m)", fontsize=12, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "baseline_table.png", dpi=150)
    print(f"\nsaved {out / 'baseline_table.md'}\nsaved {out / 'baseline_table.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
