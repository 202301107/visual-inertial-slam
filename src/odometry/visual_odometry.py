"""Monocular frame-to-frame visual odometry (ORB + essential matrix).

SCALE: a monocular camera cannot observe metric scale. Every step here returns a rotation
and a UNIT-LENGTH translation direction; no scale is invented. Chaining steps with
`chain_unit_steps` therefore gives a trajectory whose per-step lengths are all 1 (arbitrary
units). Anything metric must come from another sensor (see src/odometry/trajectory.py).

Convention for a step between previous frame p and current frame c:
    x_c = R @ x_p + t        (R, t from cv2.recoverPose, |t| = 1)
so the current camera centre, expressed in the previous camera frame, is -R.T @ t.
Camera frame: x right, y down, z forward (OpenCV).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VoStep:
    frame_idx: int
    ok: bool                  # False -> R is identity and t_dir is zero (nothing was estimated)
    R: np.ndarray             # (3, 3) previous-camera -> current-camera rotation
    t_dir: np.ndarray         # (3,) unit translation, same convention as R (zero if not ok)
    n_matches: int = 0
    n_inliers: int = 0        # points that passed the cheirality check in recoverPose
    parallax_px: float = 0.0  # median rotation-compensated feature motion (pixels, full-res scale)
    # Ratio-test matches in normalized (undistorted) coordinates, kept so the translation
    # direction can be re-estimated with an externally supplied rotation.
    n_prev: np.ndarray | None = None
    n_cur: np.ndarray | None = None

    @property
    def motion_dir_prev(self) -> np.ndarray:
        """Unit direction the camera moved, expressed in the previous camera frame."""
        return -self.R.T @ self.t_dir


class VisualOdometry:
    def __init__(self, K: np.ndarray, dist: np.ndarray, n_features: int = 2000,
                 ratio: float = 0.75, ransac_px: float = 1.0, min_matches: int = 30,
                 min_inliers: int = 20, scale: float = 1.0, fast_threshold: int = 20):
        """scale < 1 downsizes frames before ORB (K is scaled to match) to save time."""
        self.scale = scale
        self.K = K.copy()
        self.K[:2] *= scale
        self.dist = dist
        self.f = float(0.5 * (self.K[0, 0] + self.K[1, 1]))
        self.ratio, self.ransac_px = ratio, ransac_px
        self.min_matches, self.min_inliers = min_matches, min_inliers
        self._orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=fast_threshold)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._prev = None   # (keypoint pixel coords (N,2), descriptors)

    def _features(self, frame: np.ndarray):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        kps, desc = self._orb.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)
        return pts, desc

    def step(self, frame: np.ndarray, frame_idx: int) -> VoStep | None:
        """Process the next frame; returns None for the very first frame."""
        pts, desc = self._features(frame)
        prev, self._prev = self._prev, (pts, desc)
        if prev is None:
            return None
        failed = VoStep(frame_idx, False, np.eye(3), np.zeros(3))
        pts_p, desc_p = prev
        if desc is None or desc_p is None or len(desc) < 8 or len(desc_p) < 8:
            return failed

        good = []
        for pair in self._matcher.knnMatch(desc_p, desc, k=2):
            if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance:
                good.append(pair[0])
        failed.n_matches = len(good)
        if len(good) < self.min_matches:
            return failed

        p1 = pts_p[[m.queryIdx for m in good]]
        p2 = pts[[m.trainIdx for m in good]]
        n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        n2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)

        failed.n_prev, failed.n_cur = n1, n2
        E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.999, self.ransac_px / self.f)
        if E is None or E.shape != (3, 3) or mask is None:
            return failed
        # distanceThresh: the default (50 baselines) rejects almost every point at the
        # centimetre baselines of consecutive frames, so lift it.
        n_in, R, t, mask2, _ = cv2.recoverPose(E, n1, n2, np.eye(3), distanceThresh=1e9, mask=mask.copy())
        if n_in < self.min_inliers:
            failed.n_inliers = int(n_in)
            return failed

        inl = mask2.ravel() > 0
        q = (R @ np.c_[n1[inl], np.ones(inl.sum())].T).T          # previous points, derotated
        resid = n2[inl] - q[:, :2] / q[:, 2:3]
        parallax = float(np.median(np.linalg.norm(resid, axis=1)) * self.f / self.scale)
        return VoStep(frame_idx, True, R, t.ravel() / np.linalg.norm(t), len(good), int(n_in), parallax, n1, n2)


def chain_unit_steps(steps: list[VoStep]):
    """Pure VO trajectory, arbitrary scale: every ok step moves exactly 1 unit.

    Returns (R_wc, p_w): lists of camera-to-world rotations and positions in the first camera
    frame. Failed steps keep the previous pose (no motion, no rotation).
    """
    R_wc, p_w = np.eye(3), np.zeros(3)
    Rs, ps = [R_wc], [p_w]
    for s in steps:
        if s.ok:
            p_w = p_w + R_wc @ s.motion_dir_prev
            R_wc = R_wc @ s.R.T
        Rs.append(R_wc)
        ps.append(p_w)
    return Rs, ps


def translation_direction_given_rotation(n_prev: np.ndarray, n_cur: np.ndarray, R: np.ndarray,
                                         focal_px: float, thresh_px: float = 1.5,
                                         iters: int = 300, seed: int = 0):
    """Translation direction when the rotation is already known (e.g. from the gyro).

    With x_c = R x_p + t, the epipolar constraint is t . ((R n_p) x n_c) = 0 for each match, so t
    is linear in the data: RANSAC over 2-match samples, then a least-squares refit on the inliers.
    Fixing R removes the rotation/translation ambiguity that makes essential-matrix rotation
    unreliable at the centimetre baselines between consecutive frames.

    n_prev, n_cur are normalized coordinates (N, 2). Returns (t_dir unit, n_inliers, parallax_px)
    or None if it cannot be estimated. Direction only: the scale is unobservable.
    """
    if len(n_prev) < 12:
        return None
    a = (R @ np.c_[n_prev, np.ones(len(n_prev))].T).T
    x = np.c_[n_cur, np.ones(len(n_cur))]
    c = np.cross(a, x)                       # t . c_i = 0
    thr = thresh_px / focal_px
    rng = np.random.default_rng(seed)

    def residuals(t):
        l = np.cross(t, a)                   # epipolar line of each match in the current image
        return np.abs(np.sum(l * x, axis=1)) / np.maximum(np.hypot(l[:, 0], l[:, 1]), 1e-12)

    best_t, best_n = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(c), 2, replace=False)
        t = np.cross(c[i], c[j])
        nt = np.linalg.norm(t)
        if nt < 1e-12:
            continue
        n_in = int(np.sum(residuals(t / nt) < thr))
        if n_in > best_n:
            best_t, best_n = t / nt, n_in
    if best_t is None or best_n < 12:
        return None
    for _ in range(2):                       # refit on inliers
        inl = residuals(best_t) < thr
        _, _, Vt = np.linalg.svd(c[inl], full_matrices=False)
        best_t = Vt[-1]
    inl = residuals(best_t) < thr

    # Sign from cheirality: lam_c * x = lam_p * a + t must have positive depths.
    pos = 0
    for k in np.flatnonzero(inl)[:200]:
        lam, *_ = np.linalg.lstsq(np.c_[x[k], -a[k]], best_t, rcond=None)
        pos += 1 if (lam[1] > 0 and lam[0] > 0) else -1 if (lam[1] < 0 and lam[0] < 0) else 0
    if pos < 0:
        best_t = -best_t
    q = a[inl, :2] / a[inl, 2:3]
    parallax = float(np.median(np.linalg.norm(n_cur[inl] - q, axis=1)) * focal_px)
    return best_t, int(inl.sum()), parallax
