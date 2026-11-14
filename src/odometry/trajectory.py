"""Marker-free 6-DoF trajectory from monocular VO + IMU (the M4 baseline, no marker correction).

World frame: z up (from the static start gravity), x = horizontal heading of the camera's optical
axis at the first frame, origin = first camera position. Poses are camera-to-world.

Two modes
  "vio" (default)  orientation   : gyro integration (static start bias removed) with a slow
                                   accelerometer tilt correction, because essential-matrix
                                   pitch/yaw is unreliable at consecutive-frame baselines;
                   direction     : VO translation direction re-estimated with that rotation fixed,
                                   smoothed over time;
                   speed (scale) : strap-down accelerometer integration (gravity subtracted)
                                   projected on the VO direction, reset to zero when the phone is
                                   detected stationary (ZUPT). This is where metric scale comes
                                   from; it is only as good as an unbiased accel integration over
                                   a few seconds, so expect scale error and drift.
  "vo"             orientation   : chained essential-matrix rotation; every step has length 1, so
                                   the path has ARBITRARY scale. Fallback if "vio" is unusable.

Not modelled: accel bias, online gyro-bias estimation, camera-IMU time offset and lever arm.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import permutations, product

import numpy as np
from scipy.spatial.transform import Rotation

from src.acquisition.dataset_loader import RunDataset
from src.odometry.imu_preintegration import (GRAVITY, ImuDelta, exp_so3, log_so3, preintegrate, propagate,
                                             rotation_aligning, static_start_estimate)
from src.odometry.visual_odometry import (VoStep, chain_unit_steps,
                                          translation_direction_given_rotation)

Z_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class TrajectoryConfig:
    mode: str = "vio"                 # "vio", "vo" or "imu" (IMU-only dead reckoning)
    gyro_bias: str = "static"         # "static": mean gyro over the stationary start; "none"
    static_window_s: float = 1.0
    tilt_tau_s: float = 3.0           # accelerometer tilt-correction time constant
    tilt_avg_tau_s: float = 0.5       # world-frame averaging of specific force before it is used
    tilt_gate: float = 0.06           # only correct when |f_avg| is within this fraction of g
    dir_alpha: float = 0.2            # weight of each new VO direction in the smoothed direction
    min_parallax_px: float = 1.0      # VO direction ignored below this derotated parallax
    min_dir_inliers: int = 30
    zupt_gyro: float = 0.08           # rad/s, stationary if window-mean |w| is below this ...
    zupt_accel_std: float = 0.12      # ... and the accel-norm std over the window is below this
    zupt_window_s: float = 0.3
    vel_leak_tau_s: float = 2.0       # velocity decay time constant (s): bounds accel-bias drift, biases cruise speed low; 1e9 = off
    max_speed: float = 2.0            # m/s clamp, only a guard against integration blow-ups


@dataclass
class Trajectory:
    t_ns: np.ndarray                  # (N,)
    p: np.ndarray                     # (N, 3) camera position, world frame (metres in "vio")
    R_wc: np.ndarray                  # (N, 3, 3) camera-to-world rotation
    stationary: np.ndarray            # (N,) bool
    dir_ok: np.ndarray                # (N,) bool, a reliable VO direction was used at this frame
    metric: bool                      # False -> arbitrary scale ("vo" mode)
    info: dict = field(default_factory=dict)

    def save_csv(self, path) -> None:
        q = Rotation.from_matrix(self.R_wc).as_quat()            # x, y, z, w
        data = np.column_stack([self.p, q, self.stationary, self.dir_ok])
        header = "timestamp_ns,x,y,z,qx,qy,qz,qw,stationary,vo_dir_ok"
        with open(path, "w") as fh:
            fh.write(header + "\n")
            for t, row in zip(self.t_ns, data):
                fh.write(f"{int(t)}," + ",".join(f"{v:.6f}" for v in row[:7])
                         + f",{int(row[7])},{int(row[8])}\n")

    @property
    def path_length(self) -> float:
        return float(np.linalg.norm(np.diff(self.p, axis=0), axis=1).sum())


def _cube_rotations():
    out = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                M[r, c] = s
            if np.linalg.det(M) > 0:
                out.append(M)
    return out


def estimate_extrinsic(deltas: list[ImuDelta], vo_steps: list[VoStep]):
    """R_ci (camera <- IMU): the axis-aligned rotation that best matches gyro to VO rotation.

    Phone cameras are mounted axis-aligned with the sensor frame, so only the 24 signed axis
    permutations are searched. A free (Procrustes) fit is avoided on purpose: VO under-estimates
    pitch/yaw by a different factor per axis, which skews a free fit by well over 10 degrees,
    while the correct permutation still scores highest. Returns (R_ci, info) with the score
    margin over the runner-up as a confidence indicator.
    """
    pairs = []
    for d, s in zip(deltas, vo_steps):
        th_b = -log_so3(d.dR)               # points rotate by dR^T in the body frame
        if s.ok and np.linalg.norm(th_b) >= 0.004:
            pairs.append((th_b, log_so3(s.R), min(np.linalg.norm(th_b), 0.05)))
    if len(pairs) < 20:
        raise RuntimeError("Not enough rotating VO steps to estimate the camera-IMU extrinsic.")
    TB = np.array([p[0] for p in pairs])
    TC = np.array([p[1] for p in pairs])
    W = np.array([p[2] for p in pairs])
    cubes = _cube_rotations()
    scores = np.array([np.sum(W * np.sum(TC * (TB @ M.T), axis=1)) for M in cubes])
    order = np.argsort(scores)[::-1]
    best = cubes[order[0]]
    margin = float(scores[order[0]] / max(scores[order[1]], 1e-12))
    if margin < 1.3:
        warnings.warn(f"Camera-IMU extrinsic is ambiguous (best/runner-up score {margin:.2f}).")
    return best, {"extrinsic_steps": len(pairs), "extrinsic_score_margin": margin}


def _stationary_flags(ds: RunDataset, t: np.ndarray, gyro_bias: np.ndarray, cfg: TrajectoryConfig) -> np.ndarray:
    half = int(cfg.zupt_window_s * 0.5e9)
    flags = np.zeros(len(t), bool)
    for k, tk in enumerate(t):
        a = ds.accel.between(tk - half, tk + half)
        g = ds.gyro.between(tk - half, tk + half)
        if len(a) < 5 or len(g) < 5:
            continue
        w = np.linalg.norm(g.xyz - gyro_bias, axis=1).mean()
        flags[k] = w < cfg.zupt_gyro and np.linalg.norm(a.xyz, axis=1).std() < cfg.zupt_accel_std
    return flags


def build_trajectory(ds: RunDataset, vo_steps: list[VoStep], K: np.ndarray,
                     cfg: TrajectoryConfig | None = None) -> Trajectory:
    """vo_steps[k] must be the VO step from frame k to frame k+1 (len = n_frames - 1)."""
    cfg = cfg or TrajectoryConfig()
    t = ds.frame_t_ns
    n = len(t)
    n = min(n, len(vo_steps) + 1)
    t = t[:n]
    vo_steps = vo_steps[:n - 1]
    focal = float(0.5 * (K[0, 0] + K[1, 1]))
    info: dict = {"mode": cfg.mode}

    # --- static start: gravity direction, gyro bias -----------------------------------------
    f0, gb = static_start_estimate(ds.accel, ds.gyro, int(t[0]), cfg.static_window_s)
    g_mag = float(np.linalg.norm(f0))
    gyro_bias = gb if cfg.gyro_bias == "static" else np.zeros(3)
    info.update(gyro_bias_rad_s=gyro_bias.tolist(), gravity_norm=g_mag)

    # --- IMU increments per frame interval --------------------------------------------------
    deltas = []
    for k in range(n - 1):
        a, g = ds.imu_between(int(t[k]), int(t[k + 1]))
        deltas.append(preintegrate(a, g, int(t[k]), int(t[k + 1]), gyro_bias))

    R_ci, ext_info = estimate_extrinsic(deltas, vo_steps)
    R_ic = R_ci.T
    info.update(ext_info, R_ci=R_ci.round(3).tolist())

    # --- initial orientation: gravity up, heading = camera optical axis -----------------------
    R_wb0 = rotation_aligning(f0, Z_UP)
    o = R_wb0 @ R_ic @ np.array([0.0, 0, 1])
    if np.hypot(o[0], o[1]) > 0.2:
        R_wb0 = exp_so3(Z_UP * -np.arctan2(o[1], o[0])) @ R_wb0

    if cfg.mode == "imu":
        # pure strap-down dead reckoning: gyro orientation, accelerometer double integration with gravity
        # subtracted; no VO, no tilt correction, no zero-velocity updates, no velocity leak.
        g_w0 = np.array([0.0, 0.0, -g_mag])
        Rb, v, pw = R_wb0, np.zeros(3), np.zeros(3)
        R_list, p_list = [Rb], [pw]
        for d in deltas:
            Rb, v, pw = propagate(Rb, v, pw, d, g_w0)
            R_list.append(Rb)
            p_list.append(pw)
        return Trajectory(t, np.array(p_list), np.array(R_list) @ R_ic, np.zeros(n, bool), np.zeros(n, bool),
                          metric=True, info=info)

    if cfg.mode == "vo":
        Rs, ps = chain_unit_steps(vo_steps)
        R_wc0 = R_wb0 @ R_ic
        R_wc = np.array([R_wc0 @ R for R in Rs])
        p = np.array([R_wc0 @ q for q in ps])
        return Trajectory(t, p, R_wc, np.zeros(n, bool), np.array([False] + [s.ok for s in vo_steps]),
                          metric=False, info=info)

    # --- orientation: gyro + slow gravity (tilt) correction -----------------------------------
    R_wb = [R_wb0]
    f_avg = Z_UP * g_mag
    for d in deltas:
        R = R_wb[-1] @ d.dR
        a_f = min(d.dt / cfg.tilt_avg_tau_s, 1.0)
        f_avg = (1 - a_f) * f_avg + a_f * (R @ d.f_mean)
        if abs(np.linalg.norm(f_avg) - g_mag) < cfg.tilt_gate * g_mag:
            axis = np.cross(f_avg / np.linalg.norm(f_avg), Z_UP)
            R = exp_so3(axis * min(d.dt / cfg.tilt_tau_s, 1.0)) @ R
        R_wb.append(R)
    R_wb = np.array(R_wb)
    R_wc = R_wb @ R_ic

    # --- VO direction with that rotation, speed from the IMU ----------------------------------
    stationary = _stationary_flags(ds, t, gyro_bias, cfg)
    g_w = np.array([0.0, 0.0, -g_mag])
    p = np.zeros((n, 3))
    dir_ok = np.zeros(n, bool)
    v = np.zeros(3)
    d_f = None
    n_dir = 0
    for k in range(n - 1):
        d, s = deltas[k], vo_steps[k]
        v_pred = v * np.exp(-d.dt / cfg.vel_leak_tau_s) + g_w * d.dt + R_wb[k] @ d.dv

        if s.n_prev is not None:
            R_rel = R_wc[k + 1].T @ R_wc[k]
            res = translation_direction_given_rotation(s.n_prev, s.n_cur, R_rel, focal)
            if res is not None and res[2] >= cfg.min_parallax_px and res[1] >= cfg.min_dir_inliers:
                d_new = R_wc[k] @ (-R_rel.T @ res[0])
                d_f = d_new if d_f is None else (1 - cfg.dir_alpha) * d_f + cfg.dir_alpha * d_new
                d_f /= np.linalg.norm(d_f)
                dir_ok[k] = True
                n_dir += 1

        if stationary[k] or d_f is None:
            v_new = np.zeros(3)
        else:
            speed = min(max(float(d_f @ v_pred), 0.0), cfg.max_speed)
            v_new = speed * d_f
        p[k + 1] = p[k] + 0.5 * (v + v_new) * d.dt
        v = v_new

    info.update(frames_with_direction=n_dir, stationary_frames=int(stationary.sum()))
    return Trajectory(t, p, R_wc, stationary, dir_ok, metric=True, info=info)


def load_trajectory_csv(path) -> Trajectory:
    """Read a trajectory written by Trajectory.save_csv (camera-to-world poses)."""
    a = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    t = np.array(a["timestamp_ns"], dtype=np.int64)
    p = np.column_stack([a["x"], a["y"], a["z"]]).astype(float)
    R = Rotation.from_quat(np.column_stack([a["qx"], a["qy"], a["qz"], a["qw"]])).as_matrix()
    return Trajectory(t, p, R, np.array(a["stationary"], bool), np.array(a["vo_dir_ok"], bool),
                      metric=True, info={"source": str(path)})
