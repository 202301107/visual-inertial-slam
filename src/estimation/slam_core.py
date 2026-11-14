"""Fixed-map pose-graph SLAM on one run, as a function (shared by the baseline table and the ablation).

Same model as src/run_slam.py: M4 odometry edges with per-window scales, marker factors against known
marker poses, optionally explicit loop-closure edges between re-observations of the same marker.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.detection.detect_run import RunDetections
from src.estimation.graph_inputs import GraphInputs, local_scales, make_inputs, sequential_init
from src.estimation.loop_closure import LoopCandidate, find_loop_candidates
from src.estimation.pose_graph import PoseGraphResult, marker_obs_residuals, optimize_pose_graph
from src.mapping.marker_map import MapConfig, _covisible_frames
from src.odometry.trajectory import Trajectory


@dataclass
class SlamSolution:
    gi: GraphInputs
    cam_R: np.ndarray
    cam_p: np.ndarray
    result: PoseGraphResult
    loops: list[LoopCandidate]
    dropped_obs: int
    init_R: np.ndarray | None = None   # first-guess camera poses (before solving)
    init_p: np.ndarray | None = None


def solve_with_map(traj: Trajectory, det: RunDetections, markers: dict, use_ids, cfg: MapConfig | None = None,
                   loop_closure: bool = False, loop_min_gap_s: float = 10.0, max_pairs: int = 4,
                   compute_cov: bool = False, kf_limit: int | None = None) -> SlamSolution | None:
    """det must already be gated. markers: id -> 4x4 room-frame pose. Returns None if no landmark is seen.

    kf_limit: causal replay. Keep keyframes 0..kf_limit and the odometry between them, but ignore marker
    observations at keyframe kf_limit and later (so the pose at kf_limit has not yet seen a marker).
    """
    cfg = cfg or MapConfig()
    use_ids = set(use_ids) & set(markers)
    det = det.select(np.isin(det.marker_id, list(use_ids))) if use_ids else det.select(np.zeros(len(det.frame), bool))
    if not len(det.frame):
        return None
    known = {m: markers[m] for m in use_ids}
    gi = make_inputs(traj, det, cfg.kf_stride, cfg.scale_window_s, marker_ids=use_ids,
                     translation_mode=cfg.translation_mode, extra_frames=_covisible_frames(det))
    if kf_limit is not None:
        K = kf_limit + 1
        gi = replace(gi, kf_frames=gi.kf_frames[:K], kf_t_ns=gi.kf_t_ns[:K], odom_R=gi.odom_R[:K], odom_p=gi.odom_p[:K],
                     obs=[o for o in gi.obs if o.kf < kf_limit], edges=gi.edges[:K - 1], groups=gi.groups[:K - 1],
                     obs_T={k: v for k, v in gi.obs_T.items() if k[0] < kf_limit})
    if not gi.obs:
        return None
    scales, _ = local_scales(gi)
    cam_R0, cam_p0, _ = sequential_init(gi, scales, known=known, allow_new_markers=False)
    loops = find_loop_candidates(gi.obs, gi.kf_t_ns, loop_min_gap_s, max_pairs=max_pairs) if loop_closure else []
    kw = dict(odom_scale=scales, estimate_odom_scale=True, scale_groups=gi.groups, scale_walk_sigma=cfg.scale_walk_sigma,
              rel_edges=[c.edge for c in loops], max_nfev=200)
    res = optimize_pose_graph(cam_R0, cam_p0, known, gi.obs, gi.edges, compute_cov=False, **kw)
    pe, re = marker_obs_residuals(res.cam_R, res.cam_p, known, gi.obs)
    bad = (pe > cfg.outlier_trans_m) | (re > cfg.outlier_rot_deg)
    obs = gi.obs
    if bad.any():
        obs = [o for o, b in zip(gi.obs, bad) if not b]
        kw["odom_scale"] = res.odom_scales
        res = optimize_pose_graph(res.cam_R, res.cam_p, known, obs, gi.edges, compute_cov=compute_cov, **kw)
    elif compute_cov:
        res = optimize_pose_graph(res.cam_R, res.cam_p, known, obs, gi.edges, compute_cov=True,
                                  **{**kw, "odom_scale": res.odom_scales})
    return SlamSolution(gi, res.cam_R, res.cam_p, res, loops, int(bad.sum()), cam_R0, cam_p0)
