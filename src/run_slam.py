"""Marker-corrected trajectory: fuse the M4 odometry with detections of the mapped markers.

    python -m src.run_slam "data/all runs/run 1"

Needs outputs/marker_map.json (python -m src.mapping.marker_map "data/all runs/run 3") and the
run's M4 trajectory outputs/<run>_traj.csv (python -m src.run_trajectory <run>); it stops with the
command to run if either is missing. Everything is solved in the room frame (X, Y horizontal, Z up,
as surveyed in data/markers.csv) using the map's fit to that survey. Only the map's in_map markers
are used as landmarks; the floor markers are scoring points and appear in the plot only.

Writes outputs/<run>_slam.csv (keyframe poses + 1-sigma), outputs/<run>_slam.npz (poses and the full
6x6 covariances) and outputs/<run>_slam.png. The M4 file outputs/<run>_traj.csv is left untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_run import detect_run
from src.estimation.graph_inputs import local_scales, make_inputs, sequential_init
from src.estimation.pose_graph import MarkerNoise, OdomNoise, optimize_pose_graph
from src.geometry import inv_T, make_T, rot_log
from src.mapping.marker_map import DEFAULT_GT_PATH, MapConfig, MarkerMap, _gate, load_ground_truth
from src.odometry.trajectory import load_trajectory_csv


def parse_polyline(text: str) -> np.ndarray:
    return np.array([[float(v) for v in pt.split(",")] for pt in text.split(";")])


def polyline_error(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to the polyline (m)."""
    out = np.full(len(xy), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(((xy - a) @ ab) / (ab @ ab), 0, 1)
        out = np.minimum(out, np.linalg.norm(xy - (a + t[:, None] * ab), axis=1))
    return out


def register_baseline(gi, traj, cam_R, cam_p):
    """M4 trajectory placed in the room frame by its first marker-posed keyframe only.

    Yaw and translation of that single keyframe are used (the M4 frame is already gravity-aligned);
    no scale correction and nothing afterwards, so this is the honest 'before' picture.
    """
    k0 = int(min(o.kf for o in gi.obs))
    m4_R, m4_p = traj.R_wc[gi.kf_frames], traj.p[gi.kf_frames]         # the real M4 poses, M4 metres
    fwd = lambda R: np.arctan2(R[1, 2], R[0, 2])                      # heading of the optical axis
    psi = fwd(cam_R[k0]) - fwd(m4_R[k0])
    Rz = Rotation.from_euler("z", psi).as_matrix()
    p = cam_p[k0] + (Rz @ (m4_p - m4_p[k0]).T).T
    R = np.einsum("ij,njk->nik", Rz, m4_R)
    return R, p


def marker_residuals(gi, markers, cam_R, cam_p):
    """Per-observation (position error m, rotation error deg) of marker factors for given poses."""
    pe, re = [], []
    for o in gi.obs:
        T_wc = make_T(cam_R[o.kf], cam_p[o.kf])
        T_pred = inv_T(T_wc) @ markers[o.marker]
        pe.append(np.linalg.norm(T_pred[:3, 3] - o.t))
        re.append(np.degrees(np.linalg.norm(rot_log(o.R.T @ T_pred[:3, :3]))))
    return np.array(pe), np.array(re)


def plot_result(path, name, markers_room, gt, base_p, slam_p, cov, sigma_every, gt_path, kf_t):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(base_p[:, 0], base_p[:, 1], "-", color="tab:red", lw=1.4, alpha=0.8, label="M4 odometry only (no markers)")
    if gt_path is not None:
        ax.plot(gt_path[:, 0], gt_path[:, 1], "--", color="k", lw=1.6, label="taped ground-truth path")
    seg = np.stack([slam_p[:-1, :2], slam_p[1:, :2]], axis=1)
    lc = LineCollection(seg, cmap="viridis", linewidth=2.6)
    lc.set_array(0.5 * (kf_t[:-1] + kf_t[1:]))
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, shrink=0.7, label="time (s), SLAM trajectory")
    if cov is not None:
        for k in range(0, len(slam_p), sigma_every):
            C = cov[k, 3:5, 3:5]
            w, V = np.linalg.eigh(C)
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            ax.add_patch(Ellipse(slam_p[k, :2], 4 * np.sqrt(max(w[1], 0)), 4 * np.sqrt(max(w[0], 0)), angle=ang,
                                 fill=False, ec="tab:blue", lw=0.8, alpha=0.6))
        ax.plot([], [], color="tab:blue", lw=0.8, label="2-sigma position ellipse")
    ax.plot(*slam_p[0, :2], "o", color="tab:green", ms=10, label="start")
    ax.plot(*slam_p[-1, :2], "s", color="tab:red", ms=9, label="end")
    for m, pos in sorted(gt.items()):
        ax.plot(*pos["pos"][:2], "x", color="0.4", ms=6)
    for m, T in sorted(markers_room.items()):
        ax.plot(*T[:2, 3], "D", color="tab:orange", ms=8, mec="k")
        ax.annotate(str(m), T[:2, 3], textcoords="offset points", xytext=(6, 6), fontsize=10, weight="bold")
    ax.plot([], [], "D", color="tab:orange", mec="k", label="mapped marker (id)")
    ax.plot([], [], "x", color="0.4", label="surveyed marker (data/markers.csv)")
    ax.set_xlabel("X (m)  - room frame")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.08)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)
    ax.set_title(f"{name} - odometry + marker pose graph (top-down)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--map", default="outputs/marker_map.json")
    ap.add_argument("--traj", default=None, help="M4 trajectory CSV (default outputs/<run>_traj.csv)")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--gt", default=str(DEFAULT_GT_PATH))
    ap.add_argument("--stride", type=int, default=3, help="keyframe spacing in frames")
    ap.add_argument("--scale-window", type=float, default=2.0, help="odometry scale window (s)")
    ap.add_argument("--translation-mode", choices=["unit_time", "m4"], default=MapConfig.translation_mode)
    ap.add_argument("--use-ids", default=None, help="comma list of marker ids to use as landmarks "
                    "(default: the map's in_map markers, i.e. not the floor scoring points)")
    ap.add_argument("--gt-path", default=None, help='ground-truth path as "x,y;x,y;..." in room metres, for the plot/error')
    ap.add_argument("--no-cov", action="store_true")
    a = ap.parse_args(argv)

    name = Path(a.run_dir).resolve().name.strip().replace(" ", "_")
    out = Path(a.out_dir)
    traj_path = Path(a.traj) if a.traj else out / f"{name}_traj.csv"
    if not traj_path.is_file():
        raise SystemExit(f"{traj_path} not found. Build the M4 trajectory first:\n"
                         f'  python -m src.run_trajectory "{a.run_dir}"')
    traj = load_trajectory_csv(traj_path)
    det = detect_run(RunDataset(a.run_dir), cache_path=out / f"{name}_detections.npz")

    if not Path(a.map).is_file():
        raise SystemExit(f"{a.map} not found. Build the map first:\n"
                         '  python -m src.mapping.marker_map "data/all runs/run 3"')
    mmap = MarkerMap.load(a.map)
    if mmap.T_room_world is None:
        raise SystemExit("Map has no room-frame fit (T_room_world); rebuild it with src.mapping.marker_map.")
    use_ids = ({int(v) for v in a.use_ids.split(",")} if a.use_ids
               else {m for m, flag in mmap.in_map.items() if flag} or set(mmap.poses))
    markers_all = {m: mmap.T_room_world @ T for m, T in mmap.poses.items()}
    markers = {m: T for m, T in markers_all.items() if m in use_ids}
    gt = load_ground_truth(a.gt)

    cfg = MapConfig()
    det_g = _gate(det, cfg)
    gi = make_inputs(traj, det_g, a.stride, a.scale_window, marker_ids=use_ids, translation_mode=a.translation_mode)
    if not gi.obs:
        raise SystemExit("No accepted observations of the mapped markers in this run.")
    scales, s_med = local_scales(gi)
    cam_R0, cam_p0, _ = sequential_init(gi, scales, known=markers, allow_new_markers=False)
    print(f"{name}: {len(gi.kf_frames)} keyframes, {len(gi.obs)} marker observations of ids "
          f"{sorted({o.marker for o in gi.obs})}, {len(gi.edges)} odometry edges, landmarks used: {sorted(use_ids)}")

    res = optimize_pose_graph(cam_R0, cam_p0, markers, gi.obs, gi.edges, odom_scale=scales,
                              estimate_odom_scale=True, scale_groups=gi.groups, scale_walk_sigma=MapConfig.scale_walk_sigma,
                              marker_noise=MarkerNoise(), odom_noise=OdomNoise(), max_nfev=200, compute_cov=not a.no_cov)
    print(f"solve: success={res.success}, cost {res.cost_initial:.1f} -> {res.cost_final:.1f}; whitened RMS (rot, trans): "
          f"markers ({res.rms_marker[0]:.2f}, {res.rms_marker[1]:.2f}), odometry ({res.rms_odom[0]:.2f}, {res.rms_odom[1]:.2f})")
    print(f"odometry scale per {a.scale_window:g} s window ({a.translation_mode}): "
          f"min {res.odom_scales.min():.2f}, median {np.median(res.odom_scales):.2f}, max {res.odom_scales.max():.2f}")

    base_R, base_p = register_baseline(gi, traj, cam_R0, cam_p0)
    kf_t = (gi.kf_t_ns - gi.kf_t_ns[0]) * 1e-9
    pe0, re0 = marker_residuals(gi, markers, base_R, base_p)
    pe1, re1 = marker_residuals(gi, markers, res.cam_R, res.cam_p)
    print(f"\nmarker-prediction error over {len(pe0)} observations (map markers are fixed, so this is how far the "
          f"camera pose is from agreeing with what the camera saw):")
    print(f"  M4 odometry only : position median {100 * np.median(pe0):5.1f} cm, mean {100 * pe0.mean():5.1f} cm, "
          f"90% {100 * np.percentile(pe0, 90):5.1f} cm | rotation median {np.median(re0):.1f} deg")
    print(f"  pose graph       : position median {100 * np.median(pe1):5.1f} cm, mean {100 * pe1.mean():5.1f} cm, "
          f"90% {100 * np.percentile(pe1, 90):5.1f} cm | rotation median {np.median(re1):.1f} deg")

    gt_path = parse_polyline(a.gt_path) if a.gt_path else None
    def plen(p):
        return float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())
    print(f"\ntop-down path length: M4 {plen(base_p):.2f} m, pose graph {plen(res.cam_p):.2f} m"
          + (f", taped ground truth {plen(gt_path):.2f} m" if gt_path is not None else ""))
    if gt_path is not None:
        e0, e1 = polyline_error(base_p[:, :2], gt_path), polyline_error(res.cam_p[:, :2], gt_path)
        print(f"distance to the taped path: M4 mean {100 * e0.mean():.0f} cm / max {100 * e0.max():.0f} cm;  "
              f"pose graph mean {100 * e1.mean():.0f} cm / max {100 * e1.max():.0f} cm")
        print(f"end point error vs taped end {gt_path[-1]}: M4 {100 * np.linalg.norm(base_p[-1, :2] - gt_path[-1]):.0f} cm, "
              f"pose graph {100 * np.linalg.norm(res.cam_p[-1, :2] - gt_path[-1]):.0f} cm")
    sig = res.position_sigma()
    if not a.no_cov:
        print(f"position 1-sigma (from J^T J): min {np.nanmin(sig) * 100:.1f} cm, median {np.nanmedian(sig) * 100:.1f} cm, "
              f"max {np.nanmax(sig) * 100:.1f} cm")

    out.mkdir(parents=True, exist_ok=True)
    q = Rotation.from_matrix(res.cam_R).as_quat()
    csv_path, npz_path, png_path = out / f"{name}_slam.csv", out / f"{name}_slam.npz", out / f"{name}_slam.png"
    with open(csv_path, "w") as fh:
        fh.write("timestamp_ns,x,y,z,qx,qy,qz,qw,pos_sigma_m\n")
        for t, p, qq, s in zip(gi.kf_t_ns, res.cam_p, q, sig):
            fh.write(f"{int(t)}," + ",".join(f"{v:.6f}" for v in (*p, *qq)) + f",{s:.4f}\n")
    np.savez(npz_path, t_ns=gi.kf_t_ns, p=res.cam_p, R_wc=res.cam_R, cov=res.cov if res.cov is not None else np.empty(0),
             baseline_p=base_p, baseline_R=base_R, frames=gi.kf_frames,
             marker_ids=np.array(sorted(markers_all)), marker_T=np.array([markers_all[m] for m in sorted(markers_all)]))
    plot_result(png_path, name, markers_all, gt, base_p, res.cam_p, res.cov, max(len(kf_t) // 18, 1), gt_path, kf_t)
    print(f"\nsaved {csv_path}\nsaved {npz_path}\nsaved {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
