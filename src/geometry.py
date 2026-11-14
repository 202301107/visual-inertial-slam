"""Small SE(3)/SO(3) helpers shared by mapping and estimation (rotations via scipy)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def T_from_rvec_tvec(rvec, tvec) -> np.ndarray:
    return make_T(Rotation.from_rotvec(np.asarray(rvec, float).ravel()).as_matrix(),
                  np.asarray(tvec, float).ravel())


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def rot_exp(w: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(w).as_matrix()


def rotation_mean(Rs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Chordal (SVD) weighted mean of rotation matrices."""
    w = np.ones(len(Rs)) if weights is None else np.asarray(weights, float)
    U, _, Vt = np.linalg.svd(np.tensordot(w, Rs, axes=1))
    D = np.diag([1, 1, np.linalg.det(U @ Vt)])
    return U @ D @ Vt


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Similarity transform dst ~ s * R @ src + t (least squares). Returns (s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.diag([1, 1, np.linalg.det(U @ Vt)])
    R = U @ E @ Vt
    s = float(np.trace(np.diag(sig) @ E) / (S ** 2).sum() * len(src)) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s
