"""Helpers shared by the run-2 / run-5 evaluations.

Reference positions: for every frame in which a WALL marker is visible, the camera position implied by that
marker's SURVEYED pose (data/markers.csv) and the observed marker->camera pose. This is independent of the map
and of what the solver did, so it can score any trajectory (a fused one inherits only the map's own error).
Wall markers are assumed mounted rotated 180 deg in-plane (see src/run_relocalization.surveyed_marker_pose).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_run import RunDetections
from src.geometry import T_from_rvec_tvec, inv_T
from src.mapping.marker_map import load_ground_truth
from src.run_baselines import anchor_poses, register
from src.run_relocalization import surveyed_marker_pose

REASONS = ["unknown_id", "too_small", "ambiguous", "behind_camera"]


def survey_poses() -> dict:
    gt = load_ground_truth()
    return {m: surveyed_marker_pose(gt[m]) for m in gt if surveyed_marker_pose(gt[m]) is not None}


def survey_reference(det: RunDetections, surv: dict) -> dict[int, np.ndarray]:
    """frame -> camera position (room frame) from the surveyed wall-marker poses seen in that frame."""
    ref: dict[int, list] = {}
    for f, m, rv, tv in zip(det.frame, det.marker_id, det.rvec, det.tvec):
        if int(m) in surv:
            ref.setdefault(int(f), []).append((surv[int(m)] @ inv_T(T_from_rvec_tvec(rv, tv)))[:3, 3])
    return {f: np.mean(v, axis=0) for f, v in ref.items()}


def m4_in_room(traj, det: RunDetections, surv: dict):
    """M4 trajectory (full frame rate) placed in the room frame by the first-fix anchor only (yaw + translation)."""
    _, anchor = anchor_poses(det, surv, set(surv))
    return register(traj.R_wc, traj.p, anchor)


def mean_error(at, ref: dict, stride: int = 3) -> tuple[float, int]:
    """Mean top-down distance between estimate at(frame) and the reference, over every `stride`-th reference frame."""
    fr = np.array(sorted(ref))[::stride]
    e = np.array([np.linalg.norm(at(int(f))[:2] - ref[int(f)][:2]) for f in fr])
    return float(e.mean()), len(e)


def detection_summary(det: RunDetections, landmark_ids) -> dict:
    n = det.n_frames
    lm = np.isin(det.marker_id, list(landmark_ids))
    rej = {r: 0 for r in REASONS}
    if det.rej_marker is not None:
        for r in det.rej_reason:
            rej[str(r)] = rej.get(str(r), 0) + 1
    n_acc, n_rej = len(det.frame), sum(rej.values())
    return {
        "frames": n, "accepted": n_acc, "rejected": n_rej, "rej": rej,
        "frames_any": len(np.unique(det.frame)) / n, "frames_landmark": len(np.unique(det.frame[lm])) / n,
        "per_second": n_acc / (n / 30.0), "reproj_mean": float(det.reproj.mean()), "reproj_p90": float(np.percentile(det.reproj, 90)),
        "dist_mean": float(det.distance.mean()), "area_median": float(np.median(det.area)),
        "accept_ratio": n_acc / max(n_acc + n_rej, 1),
    }


def stress_indicators(ds: RunDataset, every: int = 10) -> dict:
    """Scene brightness, blur and motion intensity: what makes a run 'difficult'."""
    lum, blur = [], []
    for i, _, frame in ds.frames(step=every):
        g = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        lum.append(float(g.mean()))
        blur.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    w = np.degrees(np.linalg.norm(ds.gyro.xyz - np.median(ds.gyro.xyz[:400], axis=0), axis=1))
    a = np.linalg.norm(ds.accel.xyz, axis=1)
    return {"luminance_mean": float(np.mean(lum)), "luminance_p10": float(np.percentile(lum, 10)),
            "sharpness_median": float(np.median(blur)), "sharpness_p10": float(np.percentile(blur, 10)),
            "gyro_median_dps": float(np.median(w)), "gyro_p90_dps": float(np.percentile(w, 90)),
            "accel_std": float(a.std())}
