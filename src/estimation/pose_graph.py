"""Least-squares pose graph over camera poses (and optionally marker poses), no GTSAM.

Unknowns
    - one camera pose per keyframe (camera-to-world, T_wc),
    - optionally some marker poses (marker-to-world, T_wm); markers not listed as free are fixed,
    - optionally log-scales for the odometry translations, one per group of consecutive edges
      (monocular/IMU scale is unreliable and drifts), tied by a random-walk prior.

Factors (all residuals are whitened by their sigmas)
    odometry  between keyframes i, j:  rot : log(R_meas^T  R_i^T R_j)
                                        trans: R_i^T (p_j - p_i) - s * t_meas
              (R_meas, t_meas = relative pose measured by the M4 trajectory, expressed in frame i)
    marker    for an observation of marker m at keyframe k (marker->camera R_obs, t_obs):
                                        rot : log(R_obs^T  R_k^T R_m)
                                        trans: R_k^T (p_m - p_k) - t_obs

    prior     (optional) (p_m - pos)/sigma_pos  and  (R_m e_z - normal)/sigma_normal
    relative  (optional) same form as odometry but metric (no scale) and with its own sigmas
    scale     (only if estimated) log s_g - log s_{g-1} ~ N(0, sigma_walk), log s_0 ~ N(log s_init, sigma_prior)

Manifold handling: each pose is a fixed base pose plus a 6-vector perturbation,
R = R_base @ Exp(d_theta) (scipy Rotation), p = p_base + d_p, and the solver works on the
perturbations. The base is re-centred between solves, so rotations never leave SO(3).

Covariance: (J^T J)^-1 from the Jacobian at the solution (whitened residuals), per camera pose as
a 6x6 block ordered [rotation d_theta (camera-frame, right perturbation), translation d_p (world)].
It ignores the robust loss and assumes the noise model is right, so treat it as relative
uncertainty (it grows along odometry-only stretches and collapses at marker sightings).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


@dataclass
class MarkerNoise:
    """1-sigma of a marker pose observation: base + per_m * distance (metres / radians)."""
    t_base: float = 0.02
    t_per_m: float = 0.02
    r_base: float = 0.03
    r_per_m: float = 0.02


@dataclass
class OdomNoise:
    """1-sigma of a relative odometry measurement over one keyframe interval."""
    rot_base: float = 0.01      # rad
    rot_rel: float = 0.2        # fraction of the measured rotation angle
    t_base: float = 0.01        # m
    t_rel: float = 1.0          # fraction of the (scaled) measured step length; M4 step lengths are unreliable


@dataclass
class MarkerObs:
    kf: int                     # keyframe index
    marker: int                 # marker id
    R: np.ndarray               # (3,3) observed marker->camera rotation
    t: np.ndarray               # (3,)  observed marker->camera translation
    dist: float                 # |t|, used for the noise model


@dataclass
class MarkerPrior:
    """Soft prior on a marker's world pose, e.g. from a tape measure (position) and its facing wall."""
    marker: int
    pos: np.ndarray                       # (3,) world position of the marker centre
    sigma_pos: float = 0.10               # m
    normal: np.ndarray | None = None      # (3,) unit out-of-surface axis (marker z), None = no constraint
    sigma_normal: float = 0.15            # rad-ish, applied to the unit-vector difference


@dataclass
class RelPoseEdge:
    """Metric relative-pose constraint between two keyframes (e.g. a loop closure: the same marker seen
    at both, so T_ci_cj = T_cm(i) @ inv(T_cm(j)) is known without any map). No scale parameter."""
    i: int
    j: int
    R: np.ndarray               # (3,3) R_i^T R_j measured
    t: np.ndarray               # (3,)  R_i^T (p_j - p_i) measured, metres
    sigma_rot: float = 0.07     # rad
    sigma_trans: float = 0.05   # m


@dataclass
class OdomEdge:
    i: int
    j: int
    R: np.ndarray               # (3,3) R_i^T R_j measured
    t: np.ndarray               # (3,)  R_i^T (p_j - p_i) measured, in the odometry's own units


@dataclass
class PoseGraphResult:
    cam_R: np.ndarray                      # (K,3,3)
    cam_p: np.ndarray                      # (K,3)
    marker_poses: dict                     # id -> 4x4 (world <- marker)
    odom_scales: np.ndarray                # (G,) per-group odometry scale (G=1 if a single scale)
    cov: np.ndarray | None                 # (K,6,6) [d_theta, d_p]
    cost_initial: float
    cost_final: float
    n_marker_obs: int
    n_odom: int
    rms_marker: tuple[float, float]        # whitened (rot, trans) RMS after solving
    rms_odom: tuple[float, float]
    success: bool
    nfev: int
    rms_prior: tuple[float, float] = (float("nan"), float("nan"))   # (normal, position), whitened
    info: dict = field(default_factory=dict)

    def position_sigma(self) -> np.ndarray:
        """(K,) 1-sigma radius (sqrt of the trace of the 3x3 position covariance), metres."""
        if self.cov is None:
            return np.full(len(self.cam_p), np.nan)
        return np.sqrt(np.trace(self.cov[:, 3:, 3:], axis1=1, axis2=2))


def _batch_exp(w: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(w).as_matrix()


def _batch_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def optimize_pose_graph(cam_R, cam_p, marker_poses: dict, marker_obs: list[MarkerObs],
                        odom_edges: list[OdomEdge] = (), free_markers=(), fixed_cams=(),
                        odom_scale=1.0, estimate_odom_scale: bool = False,
                        scale_groups=None, scale_walk_sigma: float = 0.4, scale_prior_sigma: float = 1.0,
                        scale_group_runs=None, marker_priors: list[MarkerPrior] = (),
                        rel_edges: list[RelPoseEdge] = (),
                        marker_noise: MarkerNoise | None = None, odom_noise: OdomNoise | None = None,
                        loss: str = "soft_l1", f_scale: float = 2.0, max_nfev: int = 60,
                        recenter_iters: int = 3, compute_cov: bool = True,
                        verbose: int = 0) -> PoseGraphResult:
    """Solve the pose graph; see the module docstring for the model.

    cam_R/cam_p: initial camera-to-world poses per keyframe. marker_poses: id -> 4x4 world pose.
    scale_groups: (E,) int group index per odometry edge (consecutive from 0); None = one group.
    scale_group_runs: (G,) run id of each scale group; the random walk is not chained across runs.
    marker_priors: soft priors on marker poses (see MarkerPrior); only markers present in marker_poses.
    free_markers: ids whose pose is optimised (others fixed). fixed_cams: keyframes held constant
    (gauge, e.g. when markers are free and not anchored).
    """
    mn, on = marker_noise or MarkerNoise(), odom_noise or OdomNoise()
    R0 = np.array(cam_R, float).copy()
    p0 = np.array(cam_p, float).copy()
    K = len(R0)

    ids = sorted(marker_poses)
    midx = {m: i for i, m in enumerate(ids)}
    RM0 = np.array([marker_poses[m][:3, :3] for m in ids])
    pM0 = np.array([marker_poses[m][:3, 3] for m in ids])
    free_m = [midx[m] for m in ids if m in set(free_markers)]
    free_c = [k for k in range(K) if k not in set(fixed_cams)]

    obs = [o for o in marker_obs if o.marker in midx]
    N, E = len(obs), len(odom_edges)
    o_k = np.array([o.kf for o in obs], int)
    o_m = np.array([midx[o.marker] for o in obs], int)
    o_R = np.array([o.R for o in obs]).reshape(N, 3, 3)
    o_t = np.array([o.t for o in obs]).reshape(N, 3)
    o_d = np.array([o.dist for o in obs])
    s_mt, s_mr = mn.t_base + mn.t_per_m * o_d, mn.r_base + mn.r_per_m * o_d

    e_i = np.array([e.i for e in odom_edges], int)
    e_j = np.array([e.j for e in odom_edges], int)
    e_R = np.array([e.R for e in odom_edges]).reshape(E, 3, 3)
    e_t = np.array([e.t for e in odom_edges]).reshape(E, 3)
    e_ang = np.linalg.norm(_batch_log(e_R), axis=1) if E else np.zeros(0)

    cam_col = {k: 6 * n for n, k in enumerate(free_c)}
    mk_col = {m: 6 * len(free_c) + 6 * n for n, m in enumerate(free_m)}
    grp = np.zeros(E, int) if scale_groups is None else np.asarray(scale_groups, int)
    G = int(grp.max()) + 1 if (estimate_odom_scale and E) else 0
    n_x = 6 * len(free_c) + 6 * len(free_m) + G
    scale_col0 = 6 * len(free_c) + 6 * len(free_m)
    RL = len(rel_edges)
    r_i = np.array([e.i for e in rel_edges], int)
    r_j = np.array([e.j for e in rel_edges], int)
    r_R = np.array([e.R for e in rel_edges]).reshape(RL, 3, 3)
    r_t = np.array([e.t for e in rel_edges]).reshape(RL, 3)
    r_sr = np.array([e.sigma_rot for e in rel_edges], float)
    r_st = np.array([e.sigma_trans for e in rel_edges], float)
    pri = [q for q in marker_priors if q.marker in midx]
    P = len(pri)
    p_m = np.array([midx[q.marker] for q in pri], int)
    p_pos = np.array([q.pos for q in pri], float).reshape(P, 3)
    p_sp = np.array([q.sigma_pos for q in pri], float)
    p_has_n = np.array([q.normal is not None for q in pri], bool)
    p_n = np.array([q.normal if q.normal is not None else np.zeros(3) for q in pri], float).reshape(P, 3)
    p_sn = np.array([q.sigma_normal for q in pri], float)
    row_rel0 = 6 * (N + E)
    row_prior0 = row_rel0 + 6 * RL
    row_scale0 = row_prior0 + 6 * P
    n_res = row_scale0 + (G if G else 0)
    g_run = np.zeros(max(G, 1), int) if scale_group_runs is None else np.asarray(scale_group_runs, int)
    new_run = np.ones(max(G, 1), bool)                    # group starts a run -> prior to the initial scale
    new_run[1:] = g_run[1:] != g_run[:-1]
    if n_x == 0:
        raise ValueError("Nothing to optimise.")

    # sparsity: which unknowns each residual row touches
    rows, cols = [], []

    def block(row0, col0):
        for r in range(6):
            for c in range(6):
                rows.append(row0 + r)
                cols.append(col0 + c)

    for n in range(N):
        if o_k[n] in cam_col:
            block(6 * n, cam_col[o_k[n]])
        if o_m[n] in mk_col:
            block(6 * n, mk_col[o_m[n]])
    for e in range(E):
        for k in (e_i[e], e_j[e]):
            if k in cam_col:
                block(6 * N + 6 * e, cam_col[k])
        if G:
            rows += [6 * N + 6 * e + r for r in (3, 4, 5)]
            cols += [scale_col0 + grp[e]] * 3
    for e in range(RL):
        for k in (r_i[e], r_j[e]):
            if k in cam_col:
                block(row_rel0 + 6 * e, cam_col[k])
    for j in range(P):
        if p_m[j] in mk_col:
            block(row_prior0 + 6 * j, mk_col[p_m[j]])
    for g in range(G):                         # random-walk prior rows
        rows.append(row_scale0 + g)
        cols.append(scale_col0 + g)
        if g > 0 and not new_run[g]:
            rows.append(row_scale0 + g)
            cols.append(scale_col0 + g - 1)
    sparsity = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_res, n_x)).tocsr()

    s_init = np.broadcast_to(np.asarray(odom_scale, float), (max(G, 1),)).copy()
    s_base = s_init.copy()                                # per-group scale at the current base
    state = {}

    def poses(x, Rb, pb, RMb, pMb):
        d = np.zeros((K, 6))
        if free_c:
            d[free_c] = x[:6 * len(free_c)].reshape(-1, 6)
        dm = np.zeros((len(ids), 6))
        if free_m:
            dm[free_m] = x[6 * len(free_c):6 * len(free_c) + 6 * len(free_m)].reshape(-1, 6)
        R = Rb @ _batch_exp(d[:, :3])
        p = pb + d[:, 3:]
        RM = RMb @ _batch_exp(dm[:, :3])
        pM = pMb + dm[:, 3:]
        s = s_base * np.exp(x[scale_col0:scale_col0 + G]) if G else s_base
        return R, p, RM, pM, s

    def residual_parts(x, Rb, pb, RMb, pMb):
        R, p, RM, pM, s = poses(x, Rb, pb, RMb, pMb)
        parts = {}
        if N:
            Rk, pk = R[o_k], p[o_k]
            Rm, pm = RM[o_m], pM[o_m]
            Rp = np.einsum("nji,njk->nik", Rk, Rm)                      # Rk^T Rm
            tp = np.einsum("nji,nj->ni", Rk, pm - pk)                    # Rk^T (pm - pk)
            rr = _batch_log(np.einsum("nji,njk->nik", o_R, Rp)) / s_mr[:, None]
            rt = (tp - o_t) / s_mt[:, None]
            parts["marker"] = np.hstack([rr, rt])
        if E:
            Ri, Rj = R[e_i], R[e_j]
            Rp = np.einsum("nji,njk->nik", Ri, Rj)
            tp = np.einsum("nji,nj->ni", Ri, p[e_j] - p[e_i])
            t_meas = s[grp][:, None] * e_t
            sr = on.rot_base + on.rot_rel * e_ang
            st = on.t_base + on.t_rel * np.linalg.norm(t_meas, axis=1)
            rr = _batch_log(np.einsum("nji,njk->nik", e_R, Rp)) / sr[:, None]
            rt = (tp - t_meas) / st[:, None]
            parts["odom"] = np.hstack([rr, rt])
        if RL:
            Ri, Rj = R[r_i], R[r_j]
            Rp = np.einsum("nji,njk->nik", Ri, Rj)
            tp = np.einsum("nji,nj->ni", Ri, p[r_j] - p[r_i])
            rr = _batch_log(np.einsum("nji,njk->nik", r_R, Rp)) / r_sr[:, None]
            rt = (tp - r_t) / r_st[:, None]
            parts["rel"] = np.hstack([rr, rt])
        if P:
            rp = (pM[p_m] - p_pos) / p_sp[:, None]
            rn = (RM[p_m][:, :, 2] - p_n) / p_sn[:, None]
            rn[~p_has_n] = 0.0
            parts["prior"] = np.hstack([rn, rp])
        if G:
            ls = np.log(s)
            walk = np.empty(G)
            walk[0] = (ls[0] - np.log(np.median(s_init))) / scale_prior_sigma
            walk[1:] = (ls[1:] - ls[:-1]) / scale_walk_sigma
            walk[new_run] = (ls[new_run] - np.log(np.median(s_init))) / scale_prior_sigma
            parts["scale"] = walk[:, None]
        return parts

    def fun(x):
        parts = residual_parts(x, *state["base"])
        return np.concatenate([parts[k].ravel() for k in ("marker", "odom", "rel", "prior", "scale") if k in parts])

    base = (R0, p0, RM0, pM0)
    x = np.zeros(n_x)
    cost0 = None
    res = None
    nfev = 0
    for it in range(recenter_iters):
        state["base"] = base
        if cost0 is None:
            r0 = fun(np.zeros(n_x))
            cost0 = 0.5 * float(r0 @ r0)
        res = least_squares(fun, np.zeros(n_x), jac_sparsity=sparsity, loss=loss, f_scale=f_scale,
                            x_scale="jac", max_nfev=max_nfev, verbose=verbose)
        nfev += res.nfev
        Rn, pn, RMn, pMn, s_new = poses(res.x, *base)
        biggest = np.abs(res.x[:6 * len(free_c)].reshape(-1, 6)[:, :3]).max() if free_c else 0.0
        base, s_base = (Rn, pn, RMn, pMn), s_new
        if biggest < 0.2:
            break
    R_fin, p_fin, RM_fin, pM_fin = base
    state["base"] = base
    parts = residual_parts(np.zeros(n_x), *base)

    def rms(name):
        if name not in parts:
            return (float("nan"), float("nan"))
        a = parts[name]
        return (float(np.sqrt((a[:, :3] ** 2).mean())), float(np.sqrt((a[:, 3:] ** 2).mean())))

    cov = None
    if compute_cov:
        J = res.jac.tocsr() if hasattr(res.jac, "tocsr") else coo_matrix(res.jac).tocsr()
        H = (J.T @ J).toarray() + 1e-9 * np.eye(n_x)
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            Hinv = np.linalg.pinv(H)
        cov = np.zeros((K, 6, 6))
        for k, c in cam_col.items():
            cov[k] = Hinv[c:c + 6, c:c + 6]

    out_markers = {}
    for m in ids:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = RM_fin[midx[m]], pM_fin[midx[m]]
        out_markers[m] = T

    r_all = fun(np.zeros(n_x))
    return PoseGraphResult(
        cam_R=R_fin, cam_p=p_fin, marker_poses=out_markers, odom_scales=np.array(s_base), cov=cov,
        cost_initial=cost0, cost_final=0.5 * float(r_all @ r_all), n_marker_obs=N, n_odom=E,
        rms_marker=rms("marker"), rms_odom=rms("odom"), rms_prior=rms("prior"), success=bool(res.success), nfev=nfev,
        info={"n_unknowns": n_x, "dropped_obs_unknown_marker": len(marker_obs) - N})


def marker_obs_residuals(cam_R, cam_p, marker_poses: dict, obs: list[MarkerObs]):
    """(position error m, rotation error deg) of each marker observation against the given poses."""
    pe, re = np.zeros(len(obs)), np.zeros(len(obs))
    for i, o in enumerate(obs):
        Rm, pm = marker_poses[o.marker][:3, :3], marker_poses[o.marker][:3, 3]
        Rc, pc = cam_R[o.kf], cam_p[o.kf]
        t_pred = Rc.T @ (pm - pc)
        R_pred = Rc.T @ Rm
        pe[i] = np.linalg.norm(t_pred - o.t)
        re[i] = np.degrees(np.linalg.norm(Rotation.from_matrix(o.R.T @ R_pred).as_rotvec()))
    return pe, re


def rel_edge_residuals(cam_R, cam_p, edges: list[RelPoseEdge]):
    """(translation gap m, rotation gap deg) of each relative-pose edge against the given camera poses."""
    te, re = np.zeros(len(edges)), np.zeros(len(edges))
    for n, e in enumerate(edges):
        t_pred = cam_R[e.i].T @ (cam_p[e.j] - cam_p[e.i])
        R_pred = cam_R[e.i].T @ cam_R[e.j]
        te[n] = np.linalg.norm(t_pred - e.t)
        re[n] = np.degrees(np.linalg.norm(Rotation.from_matrix(e.R.T @ R_pred).as_rotvec()))
    return te, re
