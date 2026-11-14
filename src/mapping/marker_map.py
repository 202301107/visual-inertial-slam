"""Persistent marker map: world pose (4x4) of every ArUco marker, built from one run's detections.

World frame = the frame of a chosen reference marker (default id 3): its pose is the identity.
Marker frame (OpenCV/ArUco): x right, y up, z out of the marker surface toward the viewer.

Pipeline
  1. gate observations (pixel area, distance, reprojection error);
  2. chain: whenever two markers are co-visible in a frame their relative transform is
     T_ab = T_ca^-1 @ T_cb; per-pair estimates are robust-averaged over frames and propagated
     from the reference marker along the best-supported path (Dijkstra, cost 1/sqrt(n_frames));
  3. bridge + refine: co-visibility alone rarely links all markers (a handheld camera sees one
     wall at a time), so with an M4 odometry trajectory the markers are placed sequentially along
     the walk (odometry spans the short gaps) and then ALL marker poses, keyframe camera poses and
     per-window odometry scales are solved jointly (src/estimation/pose_graph.py), with the
     reference marker held fixed. Without a trajectory, only the co-visible set is refined.
Markers that cannot be linked to the reference are reported.

Check against ground truth with `compare_with_csv` (data/markers.csv, room frame).

    python -m src.mapping.marker_map "data/all runs/run 3"
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from src.detection.detect_run import RunDetections, detect_run
from src.geometry import T_from_rvec_tvec, inv_T, make_T, rot_log, rotation_mean, umeyama
from src.estimation.graph_inputs import local_scales, make_inputs, sequential_init
from src.estimation.pose_graph import MarkerObs, MarkerPrior, marker_obs_residuals, optimize_pose_graph
from src.odometry.trajectory import Trajectory, load_trajectory_csv

DEFAULT_MAP_PATH = Path("outputs/marker_map.json")
DEFAULT_GT_PATH = Path("data/markers.csv")


@dataclass
class MapConfig:
    ref_id: int = 3
    min_area_px: float = 1200.0      # stricter than the detector gate: only well-resolved markers
    max_dist_m: float = 3.5
    max_reproj_px: float = 3.0
    refine: bool = True
    max_refine_frames: int = 400     # frames with >=2 markers used in the refinement (evenly spaced)
    kf_stride: int = 3               # keyframe spacing (frames) for the odometry-bridged solve
    scale_window_s: float = 2.0      # odometry scale is re-estimated per window of this length
    translation_mode: str = "unit_time"   # see graph_inputs.make_inputs
    outlier_trans_m: float = 0.25    # after a first solve, observations further than this from their prediction
    outlier_rot_deg: float = 20.0    # ... or rotated more than this are dropped and the graph is re-solved
    scale_walk_sigma: float = 1.0    # log-speed random walk between consecutive scale groups (gaps vs viewing)
    prior_sigma_m: float = 0.10      # tape-measure prior: position 1-sigma (weak vs. the ~1-3 cm marker fixes)
    prior_sigma_normal: float = 0.15 # tape-measure prior: facing-direction 1-sigma (unit-vector difference)


@dataclass
class MarkerMap:
    reference_id: int
    poses: dict[int, np.ndarray]               # id -> 4x4 (world <- marker)
    n_obs: dict[int, int] = field(default_factory=dict)
    in_map: dict[int, bool] = field(default_factory=dict)
    T_room_world: np.ndarray | None = None     # room <- world, fitted to data/markers.csv
    info: dict = field(default_factory=dict)

    def save(self, path=DEFAULT_MAP_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        blob = {"reference_id": self.reference_id,
                "poses": {str(k): v.tolist() for k, v in sorted(self.poses.items())},
                "n_obs": {str(k): int(v) for k, v in sorted(self.n_obs.items())},
                "in_map": {str(k): bool(v) for k, v in sorted(self.in_map.items())},
                "T_room_world": None if self.T_room_world is None else self.T_room_world.tolist(),
                "info": self.info}
        Path(path).write_text(json.dumps(blob, indent=2))

    @staticmethod
    def load(path=DEFAULT_MAP_PATH) -> "MarkerMap":
        d = json.loads(Path(path).read_text())
        T = d.get("T_room_world")
        return MarkerMap(d["reference_id"], {int(k): np.array(v) for k, v in d["poses"].items()},
                         {int(k): v for k, v in d.get("n_obs", {}).items()},
                         {int(k): v for k, v in d.get("in_map", {}).items()},
                         None if T is None else np.array(T), d.get("info", {}))

    def positions(self) -> dict[int, np.ndarray]:
        return {k: v[:3, 3] for k, v in self.poses.items()}


def load_ground_truth(path=DEFAULT_GT_PATH) -> dict[int, dict]:
    """data/markers.csv -> id -> {pos (room frame, m), normal (unit, facing), in_map, wall}."""
    axes = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0),
            "+Z": (0, 0, 1), "-Z": (0, 0, -1)}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[int(r["id"])] = {"pos": np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])]),
                                 "normal": np.array(axes[r["facing"].strip()], float),
                                 "in_map": r["in_map"].strip().lower() == "yes", "wall": r["wall"]}
    return out


def _gate(det: RunDetections, cfg: MapConfig) -> RunDetections:
    keep = (det.area >= cfg.min_area_px) & (det.distance <= cfg.max_dist_m) & (det.reproj <= cfg.max_reproj_px)
    return det.select(keep)


def _frames(det: RunDetections) -> dict[int, dict[int, tuple[np.ndarray, float]]]:
    """frame -> marker id -> (T_cm 4x4 marker->camera, distance)."""
    out: dict[int, dict] = defaultdict(dict)
    for f, m, rv, tv, d in zip(det.frame, det.marker_id, det.rvec, det.tvec, det.distance):
        out[int(f)][int(m)] = (T_from_rvec_tvec(rv, tv), float(d))
    return out


def _robust_pair(Ts: list[np.ndarray], ws: np.ndarray) -> np.ndarray:
    """Weighted mean of relative transforms with one outlier-rejection pass."""
    Rs = np.array([T[:3, :3] for T in Ts])
    ts = np.array([T[:3, 3] for T in Ts])
    keep = np.ones(len(Ts), bool)
    for _ in range(2):
        R = rotation_mean(Rs[keep], ws[keep])
        t = np.average(ts[keep], axis=0, weights=ws[keep])
        dr = np.array([np.linalg.norm(rot_log(R.T @ Ri)) for Ri in Rs])
        dt = np.linalg.norm(ts - t, axis=1)
        lim_r = max(np.median(dr) + 3 * 1.48 * np.median(np.abs(dr - np.median(dr))), 0.05)
        lim_t = max(np.median(dt) + 3 * 1.48 * np.median(np.abs(dt - np.median(dt))), 0.03)
        new = (dr <= lim_r) & (dt <= lim_t)
        if new.sum() < 1 or (new == keep).all():
            break
        keep = new
    return make_T(rotation_mean(Rs[keep], ws[keep]), np.average(ts[keep], axis=0, weights=ws[keep]))


def chain_init(frames: dict, ref_id: int) -> tuple[dict[int, np.ndarray], dict]:
    """World poses of every marker reachable from ref_id through co-visibility."""
    pair_T, pair_w = defaultdict(list), defaultdict(list)
    for obs in frames.values():
        for a, b in combinations(sorted(obs), 2):
            (Ta, da), (Tb, db) = obs[a], obs[b]
            pair_T[(a, b)].append(inv_T(Ta) @ Tb)             # pose of b in a's frame
            pair_w[(a, b)].append(1.0 / (da ** 2 + db ** 2))
    edges = {k: _robust_pair(v, np.array(pair_w[k])) for k, v in pair_T.items()}
    counts = {k: len(v) for k, v in pair_T.items()}
    nbrs = defaultdict(list)
    for (a, b), T in edges.items():
        c = 1.0 / np.sqrt(counts[(a, b)])
        nbrs[a].append((b, c, T))
        nbrs[b].append((a, c, inv_T(T)))
    poses = {ref_id: np.eye(4)}
    heap = [(0.0, ref_id)]
    best = {ref_id: 0.0}
    while heap:
        cost, a = heapq.heappop(heap)
        if cost > best.get(a, np.inf):
            continue
        for b, c, T_ab in nbrs[a]:
            if cost + c < best.get(b, np.inf):
                best[b] = cost + c
                poses[b] = poses[a] @ T_ab
                heapq.heappush(heap, (cost + c, b))
    return poses, {"pair_counts": {f"{a}-{b}": n for (a, b), n in sorted(counts.items())}}


def _refine(frames: dict, poses: dict, ref_id: int, cfg: MapConfig):
    usable = sorted(f for f, o in frames.items() if sum(m in poses for m in o) >= 2)
    if len(usable) > cfg.max_refine_frames:
        usable = [usable[i] for i in np.linspace(0, len(usable) - 1, cfg.max_refine_frames).astype(int)]
    cam_R, cam_p, obs = [], [], []
    for k, f in enumerate(usable):
        m_near = min((m for m in frames[f] if m in poses), key=lambda m: frames[f][m][1])
        T_wc = poses[m_near] @ inv_T(frames[f][m_near][0])
        cam_R.append(T_wc[:3, :3])
        cam_p.append(T_wc[:3, 3])
        for m, (T_cm, d) in frames[f].items():
            if m in poses:
                obs.append(MarkerObs(k, m, T_cm[:3, :3], T_cm[:3, 3], d))
    res = optimize_pose_graph(np.array(cam_R), np.array(cam_p), poses, obs,
                              free_markers=[m for m in poses if m != ref_id], compute_cov=False)
    return res.marker_poses, {"refine_frames": len(usable), "refine_obs": len(obs),
                              "refine_cost_before": res.cost_initial, "refine_cost_after": res.cost_final,
                              "refine_rms_whitened_rot_trans": [round(v, 3) for v in res.rms_marker]}


def _build_with_odometry(det: RunDetections, traj: Trajectory, cfg: MapConfig):
    gi = make_inputs(traj, det, cfg.kf_stride, cfg.scale_window_s, translation_mode=cfg.translation_mode,
                     extra_frames=_covisible_frames(det))
    scales, s_med = local_scales(gi)
    cam_R, cam_p, markers = sequential_init(gi, scales)
    info = {"keyframes": len(gi.kf_frames), "marker_obs": len(gi.obs), "odom_scale_median_init": s_med}
    if cfg.ref_id not in markers:
        raise ValueError(f"Reference marker {cfg.ref_id} never observed at a keyframe.")
    res = optimize_pose_graph(cam_R, cam_p, markers, gi.obs, gi.edges,
                              free_markers=[m for m in markers if m != cfg.ref_id],
                              odom_scale=scales, estimate_odom_scale=True, scale_groups=gi.groups,
                              scale_walk_sigma=cfg.scale_walk_sigma, compute_cov=False)
    T_ref_inv = inv_T(res.marker_poses[cfg.ref_id])
    poses = {m: T_ref_inv @ T for m, T in res.marker_poses.items()}
    info.update(refine_cost_before=res.cost_initial, refine_cost_after=res.cost_final,
                refine_rms_whitened_rot_trans=[round(v, 3) for v in res.rms_marker],
                odom_rms_whitened_rot_trans=[round(v, 3) for v in res.rms_odom],
                odom_scales=[round(float(v), 3) for v in res.odom_scales], solver_success=res.success)
    return poses, info


def build_marker_map(det: RunDetections, cfg: MapConfig | None = None,
                     traj: Trajectory | None = None) -> MarkerMap:
    cfg = cfg or MapConfig()
    g = _gate(det, cfg)
    if cfg.ref_id not in set(g.marker_id.tolist()):
        raise ValueError(f"Reference marker {cfg.ref_id} has no accepted detections.")
    frames = _frames(g)
    chained, info = chain_init(frames, cfg.ref_id)
    seen = sorted(set(g.marker_id.tolist()))
    info["covisible_set_from_reference"] = sorted(chained)
    if traj is not None:
        poses, extra = _build_with_odometry(g, traj, cfg)
        info.update(extra)
    else:
        poses = chained
        if cfg.refine and len(poses) > 1:
            poses, rinfo = _refine(frames, poses, cfg.ref_id, cfg)
            info.update(rinfo)
    info["unreachable_ids"] = [m for m in seen if m not in poses]
    info["gate"] = {"min_area_px": cfg.min_area_px, "max_dist_m": cfg.max_dist_m, "max_reproj_px": cfg.max_reproj_px}
    n_obs = {int(m): int((g.marker_id == m).sum()) for m in seen}
    return MarkerMap(cfg.ref_id, poses, n_obs, info=info)


def fit_room_frame(mmap: MarkerMap, gt: dict, sigma_pos: float = 0.05, sigma_normal: float = 0.05):
    """Rigid transform room <- world from the in_map markers' surveyed positions and facing normals.

    Rotation: Wahba/Kabsch over the centred positions and the marker out-of-plane axes jointly
    (the in-map markers are near-coplanar, so the normals help). Returns (T_room_world, scale)
    where scale is a free-scale Umeyama fit on positions, as a metric-consistency check.
    """
    ids = [m for m in sorted(mmap.poses) if m in gt and gt[m]["in_map"]]
    if len(ids) < 3:
        raise ValueError("Need at least 3 in_map markers with ground truth for the room-frame fit.")
    P = np.array([mmap.poses[m][:3, 3] for m in ids])
    Q = np.array([gt[m]["pos"] for m in ids])
    Nw = np.array([mmap.poses[m][:3, 2] for m in ids])
    Nr = np.array([gt[m]["normal"] for m in ids])
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq) / sigma_pos ** 2 + Nw.T @ Nr / sigma_normal ** 2
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    s, _, _ = umeyama(P, Q, with_scale=True)
    return make_T(R, cq - R @ cp), s


def compare_with_csv(mmap: MarkerMap, gt: dict, T_room_world: np.ndarray | None = None) -> dict:
    """Print per-marker position/normal error vs data/markers.csv; returns summary numbers."""
    if T_room_world is None:
        T_room_world, scale = fit_room_frame(mmap, gt)
    else:
        scale = float("nan")
    rows = []
    for m in sorted(gt):
        if m not in mmap.poses:
            rows.append((m, gt[m]["in_map"], None, None))
            continue
        T = T_room_world @ mmap.poses[m]
        rows.append((m, gt[m]["in_map"], T[:3, 3] - gt[m]["pos"],
                     np.degrees(np.arccos(np.clip(T[:3, 2] @ gt[m]["normal"], -1, 1)))))
    print(f"\n{'id':>2} {'role':>9} {'est (room frame, m)':>24} {'csv (m)':>22} {'pos err':>8} {'dx dy dz (cm)':>18} {'normal err':>10}")
    for m, inmap, err, nerr in rows:
        role = "map" if inmap else "held-out"
        if err is None:
            print(f"{m:>2} {role:>9}   not placed (never co-visible with the connected set)")
            continue
        est = gt[m]["pos"] + err
        print(f"{m:>2} {role:>9} {np.array2string(est, precision=2, suppress_small=True):>24} "
              f"{np.array2string(gt[m]['pos'], precision=2):>22} {100 * np.linalg.norm(err):>6.1f}cm "
              f"{np.array2string(100 * err, precision=1):>18} {nerr:>8.1f} deg")
    out = {"scale_free_fit": scale}
    for name, flag in (("map (fit residual)", True), ("held-out", False)):
        e = [np.linalg.norm(r[2]) for r in rows if r[2] is not None and r[1] == flag]
        if e:
            out[name] = float(np.sqrt(np.mean(np.square(e))))
            print(f"RMS position error, {name}: {100 * out[name]:.1f} cm over {len(e)} markers")
    ids = [m for m in sorted(gt) if m in mmap.poses]
    d_est = [np.linalg.norm(mmap.poses[a][:3, 3] - mmap.poses[b][:3, 3]) for a, b in combinations(ids, 2)]
    d_gt = [np.linalg.norm(gt[a]["pos"] - gt[b]["pos"]) for a, b in combinations(ids, 2)]
    dd = np.array(d_est) - np.array(d_gt)
    out["pairwise_distance_rms"] = float(np.sqrt(np.mean(dd ** 2)))
    print(f"pairwise marker distances (alignment-free), all {len(dd)} pairs: RMS error {100 * out['pairwise_distance_rms']:.1f} cm, "
          f"max {100 * np.abs(dd).max():.1f} cm")
    if not np.isnan(scale):
        print(f"free-scale fit scale factor: {scale:.4f} (1.0 = metrically consistent with the survey)")
    return out


def _covisible_frames(det: RunDetections) -> np.ndarray:
    fr, cnt = np.unique(det.frame, return_counts=True)
    return fr[cnt >= 2]


@dataclass
class RunInput:
    name: str
    det: RunDetections          # already gated
    traj: Trajectory


def discover_runs(paths) -> list[Path]:
    """Run folders from the given paths; a folder holding several runs expands to its run subfolders."""
    runs: list[Path] = []
    for p in map(Path, paths):
        vids = list(p.rglob("VID_*.mp4"))
        if len(vids) == 1:
            runs.append(p)
        elif len(vids) > 1:
            runs += [c for c in sorted(p.iterdir()) if c.is_dir() and len(list(c.rglob("VID_*.mp4"))) == 1]
        else:
            raise FileNotFoundError(f"No VID_*.mp4 under {p}")
    return runs


def pooled_pair_counts(runs: list[RunInput]) -> dict:
    counts: dict = defaultdict(int)
    for r in runs:
        for obs in _frames(r.det).values():
            for a, b in combinations(sorted(obs), 2):
                counts[(a, b)] += 1
    return dict(counts)


def covisibility_components(counts: dict, ids) -> list[list[int]]:
    parent = {m: m for m in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in counts:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    comps: dict = defaultdict(list)
    for m in ids:
        comps[find(m)].append(m)
    return sorted(sorted(c) for c in comps.values())


def _best_base_name(runs: list[RunInput]) -> str:
    return max(runs, key=lambda r: (len(set(r.det.marker_id.tolist())), len(r.det.frame))).name


def build_pooled_map(runs: list[RunInput], cfg: MapConfig | None = None, gt: dict | None = None,
                     use_priors: bool = False, base_run: str | None = None) -> MarkerMap:
    """One map from several runs: every directly co-visible marker pair in ANY run is a constraint.

    Each run keeps its own keyframes and odometry (no odometry across runs); all share the marker poses.
    The start is the best single-run map (the run seeing the most distinct markers, or base_run), used
    to initialise every run's camera poses. With use_priors the whole solve is moved to the room frame
    and the in_map markers' surveyed positions/facing directions (data/markers.csv) enter as soft
    priors; the floor markers get none, so they remain held-out scoring points.
    """
    cfg = cfg or MapConfig()
    by_name = {r.name: r for r in runs}
    base_run = base_run or _best_base_name(runs)
    base = by_name[base_run]
    base_poses, _ = _build_with_odometry(base.det, base.traj, cfg)

    cam_R, cam_p, obs, edges, groups, grp_runs, scales0, run_info = [], [], [], [], [], [], [], {}
    markers = dict(base_poses)
    kf_off = grp_off = 0
    for ri, r in enumerate(runs):
        gi = make_inputs(r.traj, r.det, cfg.kf_stride, cfg.scale_window_s,
                         translation_mode=cfg.translation_mode, extra_frames=_covisible_frames(r.det))
        sc, _ = local_scales(gi)
        try:
            R0, p0, new = sequential_init(gi, sc, known=base_poses, allow_new_markers=True)
        except RuntimeError:
            print(f"  skipping {r.name}: no keyframe sees a mapped marker")
            continue
        markers.update({m: T for m, T in new.items() if m not in markers})
        cam_R.append(R0)
        cam_p.append(p0)
        obs += [MarkerObs(o.kf + kf_off, o.marker, o.R, o.t, o.dist) for o in gi.obs]
        edges += [type(e)(e.i + kf_off, e.j + kf_off, e.R, e.t) for e in gi.edges]
        groups.append(gi.groups + grp_off)
        grp_runs += [ri] * len(sc)
        scales0.append(sc)
        run_info[r.name] = {"keyframes": len(gi.kf_frames), "obs": len(gi.obs),
                            "scale_median": round(float(np.median(sc)), 3)}
        kf_off += len(gi.kf_frames)
        grp_off += len(sc)
    cam_R, cam_p = np.concatenate(cam_R), np.concatenate(cam_p)
    groups, scales0 = np.concatenate(groups), np.concatenate(scales0)

    free = [m for m in markers if m != cfg.ref_id]
    priors = []
    if use_priors:
        if gt is None:
            raise ValueError("use_priors needs the ground-truth table.")
        T_rw, _ = fit_room_frame(MarkerMap(cfg.ref_id, base_poses), gt)     # start the solve in the room frame
        cam_R = np.einsum("ij,njk->nik", T_rw[:3, :3], cam_R)
        cam_p = cam_p @ T_rw[:3, :3].T + T_rw[:3, 3]
        markers = {m: T_rw @ T for m, T in markers.items()}
        free = list(markers)                                                  # priors fix the gauge, not the ref marker
        priors = [MarkerPrior(m, gt[m]["pos"], cfg.prior_sigma_m, gt[m]["normal"], cfg.prior_sigma_normal)
                  for m in markers if m in gt and gt[m]["in_map"]]

    solve_kw = dict(free_markers=free, estimate_odom_scale=True, scale_groups=groups, scale_group_runs=grp_runs,
                    scale_walk_sigma=cfg.scale_walk_sigma, marker_priors=priors, max_nfev=100, compute_cov=False)
    res = optimize_pose_graph(cam_R, cam_p, markers, obs, edges, odom_scale=scales0, **solve_kw)
    n_before = len(obs)
    pe, re = marker_obs_residuals(res.cam_R, res.cam_p, res.marker_poses, obs)
    bad = (pe > cfg.outlier_trans_m) | (re > cfg.outlier_rot_deg)
    dropped = {}
    for o, b in zip(obs, bad):
        if b:
            dropped[o.marker] = dropped.get(o.marker, 0) + 1
    if bad.any():                                     # re-solve without the gross outliers, from the first solution
        obs = [o for o, b in zip(obs, bad) if not b]
        res = optimize_pose_graph(res.cam_R, res.cam_p, res.marker_poses, obs, edges,
                                  odom_scale=res.odom_scales, **solve_kw)
    T_ref_inv = inv_T(res.marker_poses[cfg.ref_id])
    poses = {m: T_ref_inv @ T for m, T in res.marker_poses.items()}
    seen = sorted({o.marker for o in obs})
    info = {"runs": run_info, "base_run": base_run, "keyframes": kf_off, "marker_obs": len(obs),
            "pooled_pair_counts": {f"{a}-{b}": n for (a, b), n in sorted(pooled_pair_counts(runs).items())},
            "use_priors": use_priors, "prior_sigma_m": cfg.prior_sigma_m if use_priors else None,
            "refine_cost_before": res.cost_initial, "refine_cost_after": res.cost_final,
            "refine_rms_whitened_rot_trans": [round(v, 3) for v in res.rms_marker],
            "odom_rms_whitened_rot_trans": [round(v, 3) for v in res.rms_odom],
            "prior_rms_whitened_normal_pos": [round(v, 3) for v in res.rms_prior],
            "outliers_dropped": {"total": n_before - len(obs), "by_marker": dropped},
            "solver_success": res.success, "unreachable_ids": [m for m in seen if m not in poses]}
    n_obs = {int(m): sum(int((r.det.marker_id == m).sum()) for r in runs) for m in seen}
    return MarkerMap(cfg.ref_id, poses, n_obs, info=info)


def error_table(mmap: MarkerMap, gt: dict) -> dict:
    """Per-marker position error (m), normal error (deg) and RMS summaries vs the survey."""
    T_rw, scale = fit_room_frame(mmap, gt)
    pos, nrm = {}, {}
    for m in sorted(gt):
        if m in mmap.poses:
            T = T_rw @ mmap.poses[m]
            pos[m] = float(np.linalg.norm(T[:3, 3] - gt[m]["pos"]))
            nrm[m] = float(np.degrees(np.arccos(np.clip(T[:3, 2] @ gt[m]["normal"], -1, 1))))

    def rms(ids):
        v = [pos[m] ** 2 for m in ids if m in pos]
        return float(np.sqrt(np.mean(v))) if v else float("nan")

    ids = sorted(pos)
    dd = [np.linalg.norm(mmap.poses[a][:3, 3] - mmap.poses[b][:3, 3]) - np.linalg.norm(gt[a]["pos"] - gt[b]["pos"])
          for a, b in combinations(ids, 2)]
    return {"pos": pos, "normal": nrm, "scale": scale,
            "rms_map": rms([m for m in gt if gt[m]["in_map"]]), "rms_held": rms([m for m in gt if not gt[m]["in_map"]]),
            "pair_rms": float(np.sqrt(np.mean(np.square(dd)))), "pair_max": float(np.abs(dd).max())}


def print_comparison(tables: dict, gt: dict) -> None:
    labels = list(tables)
    w = 22
    print("\nper-marker error vs data/markers.csv: position (cm) / facing-direction (deg), after a rigid fit to the in_map markers")
    print(f"{'id':>2} {'role':>9}  " + " ".join(f"{l:>{w}}" for l in labels))
    for m in sorted(gt):
        cells = []
        for l in labels:
            t = tables[l]
            cells.append(f"{100 * t['pos'][m]:5.1f} / {t['normal'][m]:5.1f}" if m in t["pos"] else "not placed")
        print(f"{m:>2} {'map' if gt[m]['in_map'] else 'held-out':>9}  " + " ".join(f"{c:>{w}}" for c in cells))
    rows = (("rms_map", "RMS position error, in_map markers 0-6 (cm)", 100), ("rms_held", "RMS position error, held-out 7-9 (cm)", 100),
            ("pair_rms", "RMS pairwise-distance error, all pairs (cm)", 100), ("pair_max", "max pairwise-distance error (cm)", 100),
            ("scale", "free-scale fit factor (1.0 = metric)", 1))
    for key, name, mult in rows:
        print(f"{name:<46} " + " ".join(f"{tables[l][key] * mult:>{w}.{1 if mult == 100 else 3}f}" for l in labels))


def main(argv=None) -> int:
    from src.acquisition.dataset_loader import RunDataset
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", help='run folder(s), or a folder of runs such as "data/all runs"')
    ap.add_argument("--ref-id", type=int, default=MapConfig.ref_id)
    ap.add_argument("--out", default=str(DEFAULT_MAP_PATH),
                    help="pure-SLAM map; the prior-assisted map goes to <out stem>_prior.json")
    ap.add_argument("--gt", default=str(DEFAULT_GT_PATH))
    ap.add_argument("--min-area", type=float, default=MapConfig.min_area_px)
    ap.add_argument("--max-dist", type=float, default=MapConfig.max_dist_m)
    ap.add_argument("--translation-mode", choices=["unit_time", "m4"], default=MapConfig.translation_mode)
    ap.add_argument("--base-run", default=None, help="run (folder name) used to initialise; default: most markers seen")
    ap.add_argument("--use-priors", action="store_true",
                    help="ALSO build a map with data/markers.csv as soft priors (in_map markers only); off by default")
    ap.add_argument("--prior-sigma", type=float, default=MapConfig.prior_sigma_m, help="prior position 1-sigma, m")
    a = ap.parse_args(argv)

    out = Path(a.out)
    cfg = MapConfig(ref_id=a.ref_id, min_area_px=a.min_area, max_dist_m=a.max_dist,
                    translation_mode=a.translation_mode, prior_sigma_m=a.prior_sigma)
    gt = load_ground_truth(a.gt)
    runs, missing = [], []
    for d in discover_runs(a.runs):
        name = d.resolve().name.strip().replace(" ", "_")
        traj_path = out.parent / f"{name}_traj.csv"
        if not traj_path.is_file():
            missing.append(f'python -m src.run_trajectory "{d}"')
            continue
        det = detect_run(RunDataset(d), cache_path=out.parent / f"{name}_detections.npz")
        runs.append(RunInput(name, _gate(det, cfg), load_trajectory_csv(traj_path)))
    if missing:
        raise SystemExit("M4 trajectories are missing for some runs; build them first:\n  " + "\n  ".join(missing))
    print(f"runs pooled: {[r.name for r in runs]}")
    for r in runs:
        ids, c = np.unique(r.det.marker_id, return_counts=True)
        print(f"  {r.name}: gated detections per id {dict(zip(ids.tolist(), c.tolist()))}")

    base_name = a.base_run or _best_base_name(runs)
    single = [r for r in runs if r.name == base_name]
    c_all, c_one = pooled_pair_counts(runs), pooled_pair_counts(single)
    print(f"\ndirectly co-visible marker pairs: pooled {len(c_all)} pairs / {sum(c_all.values())} frames; "
          f"{base_name} alone {len(c_one)} pairs / {sum(c_one.values())} frames")
    print(f"co-visibility groups, {base_name} alone: {covisibility_components(c_one, sorted(gt))}")
    print(f"co-visibility groups, pooled:       {covisibility_components(c_all, sorted(gt))}")
    print("pooled pair counts (frames):", {f"{p}-{q}": n for (p, q), n in sorted(c_all.items())})

    maps: dict[str, MarkerMap] = {}
    print(f"\n[1/3] single-run map ({base_name} only, odometry bridging)")
    b_poses, b_info = _build_with_odometry(single[0].det, single[0].traj, cfg)
    maps[f"{base_name} only"] = MarkerMap(cfg.ref_id, b_poses, info=b_info)
    print("[2/3] pooled pure-SLAM map (all runs, no priors)")
    maps["pooled pure SLAM"] = build_pooled_map(runs, cfg, gt, False, base_name)
    if a.use_priors:
        print(f"[3/3] pooled prior-assisted map (data/markers.csv priors, sigma {cfg.prior_sigma_m * 100:.0f} cm)")
        maps["pooled + priors"] = build_pooled_map(runs, cfg, gt, True, base_name)

    for label, mm in maps.items():
        for m in gt:
            mm.in_map[m] = gt[m]["in_map"]
        mm.T_room_world, _ = fit_room_frame(mm, gt)
        mm.info["label"] = label
    pure = maps["pooled pure SLAM"]
    pi = pure.info
    print(f"\npooled pure map: {pi['keyframes']} keyframes, {pi['marker_obs']} obs; solve cost {pi['refine_cost_before']:.0f} -> "
          f"{pi['refine_cost_after']:.0f}; whitened RMS markers {pi['refine_rms_whitened_rot_trans']}, odometry {pi['odom_rms_whitened_rot_trans']}")
    print("per-run:", pi["runs"], "| outlier observations dropped:", pi["outliers_dropped"])
    if a.use_priors:
        q = maps["pooled + priors"].info
        print(f"prior-assisted map: whitened prior RMS (normal, position) {q['prior_rms_whitened_normal_pos']}, "
              f"markers {q['refine_rms_whitened_rot_trans']}")
    print_comparison({l: error_table(m, gt) for l, m in maps.items()}, gt)

    pure.save(out)
    print(f"\nsaved {out}  (pure SLAM, pooled over {len(runs)} runs)")
    if a.use_priors:
        pp = out.with_name(out.stem + "_prior.json")
        maps["pooled + priors"].save(pp)
        print(f"saved {pp}  (prior-assisted; priors on in_map markers only, sigma {cfg.prior_sigma_m * 100:.0f} cm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
