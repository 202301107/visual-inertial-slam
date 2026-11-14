"""Turn an M4 trajectory plus marker detections into pose-graph inputs and a first guess.

Keyframes are every `stride`-th frame. Odometry edges come from consecutive keyframes of the M4
trajectory (in its own, wrongly scaled, units); marker observations are kept only at keyframes.
`sequential_init` walks the keyframes in order: where a marker with a known pose is visible the
camera pose follows from it, otherwise the previous pose is advanced by the (scaled) odometry,
and markers seen for the first time are placed from the current camera pose. The odometry only
has to bridge the short stretches with no mapped marker in view.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.detection.detect_run import RunDetections
from src.estimation.pose_graph import MarkerObs, OdomEdge
from src.geometry import T_from_rvec_tvec, inv_T, make_T
from src.odometry.trajectory import Trajectory


@dataclass
class GraphInputs:
    kf_frames: np.ndarray            # (K,) frame index of each keyframe
    kf_t_ns: np.ndarray              # (K,)
    odom_R: np.ndarray               # (K,3,3) M4 camera-to-world rotation
    odom_p: np.ndarray               # (K,3) M4 camera position, M4 units
    obs: list[MarkerObs]
    edges: list[OdomEdge]
    groups: np.ndarray               # (E,) odometry scale-group per edge
    obs_T: dict                      # (kf, marker) -> 4x4 marker->camera


def make_inputs(traj: Trajectory, det: RunDetections, stride: int = 3,
                group_seconds: float = 2.0, marker_ids=None,
                translation_mode: str = "unit_time", extra_frames=None,
                group_by_visibility: bool = True) -> GraphInputs:
    """translation_mode:
      "m4"        odometry steps keep the M4 step lengths (scaled by the solved odometry scale);
      "unit_time" only the M4 step DIRECTION is kept; the length is (odometry scale) x (time
                  step), i.e. a constant speed per scale window and zero while M4 flags the phone
                  stationary. M4's IMU-derived step lengths are too unreliable to trust.
    """
    n = len(traj.t_ns)
    kf_set = set(range(0, n, stride)) | {n - 1}
    if extra_frames is not None:                  # e.g. frames where two markers are seen together
        kf_set |= {int(f) for f in extra_frames if 0 <= int(f) < n}
    kf = np.array(sorted(kf_set))
    kf_of = {int(f): i for i, f in enumerate(kf)}
    obs, obs_T = [], {}
    for f, m, rv, tv, d in zip(det.frame, det.marker_id, det.rvec, det.tvec, det.distance):
        if int(f) in kf_of and (marker_ids is None or int(m) in marker_ids):
            T = T_from_rvec_tvec(rv, tv)
            obs.append(MarkerObs(kf_of[int(f)], int(m), T[:3, :3], T[:3, 3], float(d)))
            obs_T[(kf_of[int(f)], int(m))] = T
    R, p = traj.R_wc[kf], traj.p[kf]
    t = traj.t_ns[kf]
    visible = np.zeros(len(kf), bool)
    for o in obs:
        visible[o.kf] = True
    t_rel = [R[i].T @ (p[i + 1] - p[i]) for i in range(len(kf) - 1)]
    if translation_mode == "unit_time":
        dt = np.diff(t) * 1e-9
        still = traj.stationary[kf]
        dirs, last = [], None
        for i, v in enumerate(t_rel):
            nv = np.linalg.norm(v)
            if nv > 1e-4:
                last = v / nv
            # M4 also reports zero translation when its IMU speed estimate was clamped, not only when
            # the phone is still: keep zero only where M4 flags stationary, else reuse the last direction.
            moving = not (still[i] and still[i + 1])
            dirs.append(last if (moving and last is not None) else None)
        for i in range(len(dirs) - 1, -1, -1):             # leading gap: borrow the next direction
            if dirs[i] is None and not (still[i] and still[i + 1]):
                dirs[i] = next((d for d in dirs[i + 1:] if d is not None), None)
        t_rel = [d * dt[i] if d is not None else np.zeros(3) for i, d in enumerate(dirs)]
        p = np.vstack([np.zeros(3), np.cumsum([R[i] @ t_rel[i] for i in range(len(kf) - 1)], axis=0)])
    elif translation_mode != "m4":
        raise ValueError(f"unknown translation_mode {translation_mode!r}")
    edges = [OdomEdge(i, i + 1, R[i].T @ R[i + 1], t_rel[i]) for i in range(len(kf) - 1)]
    if group_by_visibility:
        # A new scale group starts when the walk enters/leaves a marker-free gap, or after
        # group_seconds: people linger near markers but cross the room between them far faster
        # (measured 4-10x), so a gap must not inherit the speed of the stretch next to it.
        gap = ~(visible[:-1] | visible[1:])
        groups = np.zeros(len(kf) - 1, int)
        g, t_start = 0, t[0]
        for e in range(1, len(kf) - 1):
            if gap[e] != gap[e - 1] or (t[e] - t_start) * 1e-9 > group_seconds:
                g += 1
                t_start = t[e]
            groups[e] = g
    else:
        groups = np.minimum(((t[:-1] - t[0]) * 1e-9 / group_seconds).astype(int), 10 ** 6)
        _, groups = np.unique(groups, return_inverse=True)
    return GraphInputs(kf, t, R, p, obs, edges, groups, obs_T)


def local_scales(gi: GraphInputs, min_disp_m: float = 0.25, max_dt_s: float = 3.0):
    """Odometry scale per group (marker-metric displacement / M4 displacement) from marker sightings.

    Uses pairs of keyframes that see the same marker: the camera displacement between them is
    known metrically from the marker poses alone, no map needed. Returns (per-group scale array,
    global median). Groups without usable pairs get the global median.
    """
    by_marker = defaultdict(list)
    for (k, m), T in gi.obs_T.items():
        by_marker[m].append((k, -T[:3, :3].T @ T[:3, 3]))        # camera centre in the marker frame
    G = int(gi.groups.max()) + 1 if len(gi.groups) else 1
    ratios = defaultdict(list)
    for m, lst in by_marker.items():
        lst.sort(key=lambda x: x[0])
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                ka, pa = lst[a]
                kb, pb = lst[b]
                if (gi.kf_t_ns[kb] - gi.kf_t_ns[ka]) * 1e-9 > max_dt_s:
                    break
                d_m = np.linalg.norm(pb - pa)
                d_o = np.linalg.norm(gi.odom_p[kb] - gi.odom_p[ka])
                if d_m >= min_disp_m and d_o > 0.02:
                    ratios[int(gi.groups[min(ka, len(gi.groups) - 1)])].append(d_m / d_o)
    allr = [r for v in ratios.values() for r in v]
    g_med = float(np.median(allr)) if allr else 0.6
    per = np.array([float(np.median(ratios[g])) if len(ratios[g]) >= 3 else g_med for g in range(G)])
    return np.clip(per, 0.05, 30.0), g_med


def sequential_init(gi: GraphInputs, scales: np.ndarray, known: dict | None = None,
                    allow_new_markers: bool = True, anchor_first: int | None = None):
    """First guess of camera poses (and new marker poses) in the frame of `known` markers.

    known: id -> 4x4 world pose (may be empty/None when mapping; then the world is the
    gravity-aligned odometry frame, scaled, anchored at the first sighting). Returns (cam_R, cam_p, marker_poses).
    """
    K = len(gi.kf_frames)
    markers = dict(known or {})
    by_kf = defaultdict(dict)
    for (k, m), T in gi.obs_T.items():
        by_kf[k][m] = T
    cam = [None] * K

    def scale_at(k):
        return float(scales[gi.groups[min(k, len(gi.groups) - 1)]])

    def propagate(k_from, k_to):               # advance pose k_from -> k_to along the odometry
        T = cam[k_from]
        if k_to > k_from:
            e = gi.edges[k_from]
            return make_T(T[:3, :3] @ e.R, T[:3, 3] + T[:3, :3] @ (scale_at(k_from) * e.t))
        e = gi.edges[k_to]
        R_to = T[:3, :3] @ e.R.T
        return make_T(R_to, T[:3, 3] - R_to @ (scale_at(k_to) * e.t))

    first = None
    for k in range(K):
        seen = by_kf.get(k, {})
        known_here = [m for m in seen if m in markers]
        if known_here:
            m = min(known_here, key=lambda m: np.linalg.norm(seen[m][:3, 3]))
            cam[k] = markers[m] @ inv_T(seen[m])
        elif k > 0 and cam[k - 1] is not None:
            cam[k] = propagate(k - 1, k)
        elif seen and allow_new_markers and not markers:     # very first marker defines the world
            m0 = min(seen, key=lambda m: np.linalg.norm(seen[m][:3, 3]))
            cam[k] = make_T(gi.odom_R[k], scale_at(k) * gi.odom_p[k])   # world = scaled odometry frame (z up)
            markers[m0] = cam[k] @ seen[m0]
        if cam[k] is not None:
            if first is None:
                first = k
            if allow_new_markers:
                for m, T in seen.items():
                    if m not in markers:
                        markers[m] = cam[k] @ T
    if first is None:
        raise RuntimeError("No keyframe observes a mapped marker; cannot initialise.")
    for k in range(first - 1, -1, -1):                       # before the first sighting
        cam[k] = propagate(k + 1, k)
    return (np.array([c[:3, :3] for c in cam]), np.array([c[:3, 3] for c in cam]), markers)
