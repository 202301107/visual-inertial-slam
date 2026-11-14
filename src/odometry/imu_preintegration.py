"""Strap-down IMU integration between consecutive camera frames.

`preintegrate` integrates the gyro and the accelerometer over one frame interval and returns
increments expressed in the IMU body frame at the START of the interval, so they do not depend
on the (unknown) world orientation or on gravity. `propagate` composes them onto a world-frame
state and subtracts gravity at that point.

Conventions
    - Body frame: the phone's sensor axes as logged (accelerometer reports specific force,
      i.e. it reads +g "up" at rest).
    - World frame: z up, gravity g_w = (0, 0, -9.81).
    - R_wb maps body vectors to world vectors; dR = R_{b_k b_{k+1}}, so R_wb[k+1] = R_wb[k] @ dR.

NOT modelled: accelerometer/gyro bias (callers may pass a constant bias, e.g. the mean gyro
reading over a stationary start), scale/misalignment errors, camera-IMU lever arm and time offset.
Open-loop double integration of a phone-grade accelerometer drifts quickly; the integrated
velocity/position are only trustworthy over short intervals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.acquisition.dataset_loader import ImuData

GRAVITY = 9.81


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rotation matrix for rotation vector w (Rodrigues)."""
    th = float(np.linalg.norm(w))
    K = skew(w)
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + (np.sin(th) / th) * K + ((1 - np.cos(th)) / th ** 2) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vector of rotation matrix R."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-7:
        return 0.5 * v
    if np.pi - th < 1e-5:   # near 180 deg: sin(th) ~ 0, use the symmetric part
        A = (R + np.eye(3)) / 2
        axis = np.sqrt(np.maximum(np.diag(A), 0))
        k = int(np.argmax(axis))
        axis = A[:, k] / max(axis[k], 1e-12)
        return th * axis / np.linalg.norm(axis)
    return th / (2 * np.sin(th)) * v


@dataclass
class ImuDelta:
    dt: float              # interval length, s
    dR: np.ndarray         # (3,3) R_{b_k b_{k+1}}
    dv: np.ndarray         # (3,) integral of R_{b_k b(t)} f dt   (specific force, gravity NOT removed)
    dp: np.ndarray         # (3,) double integral of the same
    f_mean: np.ndarray     # (3,) mean specific force over the interval (bias-corrected, body frame)
    w_mean: np.ndarray     # (3,) mean angular rate over the interval (bias-corrected, body frame)
    n_accel: int
    n_gyro: int


def preintegrate(accel: ImuData, gyro: ImuData, t0_ns: int, t1_ns: int,
                 gyro_bias: np.ndarray | None = None,
                 accel_bias: np.ndarray | None = None) -> ImuDelta:
    """Integrate IMU samples for t0_ns <= t < t1_ns (pass the output of RunDataset.imu_between).

    Gyro samples are held piecewise-constant from their timestamp; the accelerometer is linearly
    interpolated at each gyro sub-step midpoint. If the interval holds no gyro sample the rotation
    is the identity; if it holds no accel sample dv and dp are zero.
    """
    gb = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, float)
    ab = np.zeros(3) if accel_bias is None else np.asarray(accel_bias, float)
    dt_total = (t1_ns - t0_ns) * 1e-9

    if len(gyro):
        times = np.concatenate(([t0_ns], gyro.t_ns, [t1_ns]))
        omegas = np.vstack((gyro.xyz[0], gyro.xyz)) - gb
    else:
        times, omegas = np.array([t0_ns, t1_ns]), np.zeros((1, 3))
    tsec = (times - t0_ns) * 1e-9
    if len(accel):
        ta = (accel.t_ns - t0_ns) * 1e-9
        fa = accel.xyz - ab

    R, v, p = np.eye(3), np.zeros(3), np.zeros(3)
    for j in range(len(tsec) - 1):
        dt = tsec[j + 1] - tsec[j]
        if dt <= 0:
            continue
        w = omegas[j]
        if len(accel):
            tm = 0.5 * (tsec[j] + tsec[j + 1])
            f = np.array([np.interp(tm, ta, fa[:, c]) for c in range(3)])
            a_b = R @ exp_so3(w * dt / 2) @ f
            p += v * dt + 0.5 * a_b * dt * dt
            v += a_b * dt
        R = R @ exp_so3(w * dt)

    f_mean = fa.mean(0) if len(accel) else np.zeros(3)
    w_mean = (omegas.mean(0) if len(gyro) else np.zeros(3))
    return ImuDelta(dt_total, R, v, p, f_mean, w_mean, len(accel), len(gyro))


def propagate(R_wb: np.ndarray, v_w: np.ndarray, p_w: np.ndarray, d: ImuDelta,
              g_w: np.ndarray = np.array([0.0, 0.0, -GRAVITY])):
    """Compose an ImuDelta onto a world-frame state; gravity is subtracted here.

    Returns (R_wb', v_w', p_w').
    """
    v_new = v_w + g_w * d.dt + R_wb @ d.dv
    p_new = p_w + v_w * d.dt + 0.5 * g_w * d.dt ** 2 + R_wb @ d.dp
    return R_wb @ d.dR, v_new, p_new


def static_start_estimate(accel: ImuData, gyro: ImuData, t_start_ns: int, duration_s: float = 1.0):
    """Mean specific force and mean gyro reading over a window assumed stationary.

    Returns (f_mean, gyro_bias). The gyro mean of a stationary phone is its bias (the earth rate
    is negligible at this grade). Also the only source of the initial gravity direction.
    """
    t1 = t_start_ns + int(duration_s * 1e9)
    a, g = accel.between(t_start_ns, t1), gyro.between(t_start_ns, t1)
    if len(a) < 5 or len(g) < 5:
        raise ValueError("Not enough IMU samples in the static start window.")
    return a.xyz.mean(0), g.xyz.mean(0)


def rotation_aligning(u: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal rotation R with R @ u = target (both treated as directions)."""
    u = u / np.linalg.norm(u)
    target = target / np.linalg.norm(target)
    axis = np.cross(u, target)
    s, c = np.linalg.norm(axis), float(u @ target)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
        return exp_so3(np.pi * perp / np.linalg.norm(perp))
    return exp_so3(axis / s * np.arctan2(s, c))
