"""Loop closure on a closed-loop run (run 3): residual before/after, and the effect on the marker map.

    python -m src.run_loop_closure "data/all runs/run 3"

Variants solved on the same keyframes/odometry (marker poses free, reference marker fixed):
  initial     odometry-bridged first guess, nothing solved
  A  no LC    late revisits of a marker get their own landmark id and nothing ties them to the early
              visit: this is what the graph does if the revisit is not recognised
  B  +LC      same, plus explicit relative-pose edges between early and late keyframes of the same marker
  C  shared   the normal map: one landmark per marker, so both visits constrain the same pose
              (loop closure by data association; the explicit edge would be redundant here)
Reported: loop-edge gap (translation m / rotation deg between the two estimates of the same place, from
the camera poses of each variant), landmark-copy gap, and the map error against data/markers.csv.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_run import detect_run
from src.estimation.graph_inputs import local_scales, make_inputs, sequential_init
from src.estimation.loop_closure import find_loop_candidates, relabel_revisits
from src.estimation.pose_graph import marker_obs_residuals, optimize_pose_graph, rel_edge_residuals
from src.geometry import inv_T, rot_log
from src.mapping.marker_map import (MapConfig, MarkerMap, _covisible_frames, _gate, error_table, fit_room_frame,
                                    load_ground_truth)
from src.odometry.trajectory import load_trajectory_csv


def solve_variant(gi, obs, rel_edges, cfg, ref_id):
    """Sequential init + joint solve (+ one outlier pass). Returns (init, result, dropped)."""
    giv = replace(gi, obs=obs, obs_T={(o.kf, o.marker): np.vstack([np.hstack([o.R, o.t[:, None]]), [0, 0, 0, 1]]) for o in obs})
    scales, _ = local_scales(giv)
    cam_R, cam_p, markers = sequential_init(giv, scales)
    init = (cam_R, cam_p, markers)
    kw = dict(free_markers=[m for m in markers if m != ref_id], estimate_odom_scale=True, scale_groups=gi.groups,
              scale_walk_sigma=cfg.scale_walk_sigma, rel_edges=rel_edges, max_nfev=100, compute_cov=False)
    res = optimize_pose_graph(cam_R, cam_p, markers, obs, gi.edges, odom_scale=scales, **kw)
    pe, re = marker_obs_residuals(res.cam_R, res.cam_p, res.marker_poses, obs)
    bad = (pe > cfg.outlier_trans_m) | (re > cfg.outlier_rot_deg)
    if bad.any():
        keep = [o for o, b in zip(obs, bad) if not b]
        res = optimize_pose_graph(res.cam_R, res.cam_p, res.marker_poses, keep, gi.edges, odom_scale=res.odom_scales, **kw)
    return init, res, int(bad.sum())


def pose_gap(Ta, Tb):
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), float(np.degrees(np.linalg.norm(rot_log(Ta[:3, :3].T @ Tb[:3, :3]))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--min-gap", type=float, default=10.0, help="seconds between visits for a re-observation to count as a loop")
    ap.add_argument("--max-pairs", type=int, default=4, help="loop-closure edges per revisited marker")
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    name = Path(a.run_dir).resolve().name.strip().replace(" ", "_")
    cfg, gt = MapConfig(), load_ground_truth()
    det = _gate(detect_run(RunDataset(a.run_dir), cache_path=out / f"{name}_detections.npz"), cfg)
    traj = load_trajectory_csv(out / f"{name}_traj.csv")
    gi = make_inputs(traj, det, cfg.kf_stride, cfg.scale_window_s, translation_mode=cfg.translation_mode,
                     extra_frames=_covisible_frames(det))

    cands = find_loop_candidates(gi.obs, gi.kf_t_ns, a.min_gap, max_pairs=a.max_pairs)
    if not cands:
        raise SystemExit(f"No marker is re-observed after a gap of >= {a.min_gap:g} s; nothing to close.")
    t0 = gi.kf_t_ns[0]
    print(f"{name}: {len(gi.kf_frames)} keyframes, {len(gi.obs)} marker observations")
    print("loop-closure candidates (same marker re-observed after a circuit):")
    for c in cands:
        print(f"  marker {c.marker}: keyframe {c.kf_i} (t={(gi.kf_t_ns[c.kf_i] - t0) * 1e-9:.1f} s) <-> {c.kf_j} "
              f"(t={(gi.kf_t_ns[c.kf_j] - t0) * 1e-9:.1f} s), gap {c.gap_s:.0f} s, measured relative translation "
              f"{np.linalg.norm(c.edge.t):.2f} m")
    edges = [c.edge for c in cands]

    obs_split, copies = relabel_revisits(gi.obs, gi.kf_t_ns, a.min_gap)
    print(f"landmark copies created for the no-loop-closure control: {copies}\n")

    results = {}
    print("solving: A no loop closure ...")
    initA, resA, dA = solve_variant(gi, obs_split, [], cfg, cfg.ref_id)
    print("solving: B explicit loop closure ...")
    _, resB, dB = solve_variant(gi, obs_split, edges, cfg, cfg.ref_id)
    print("solving: C shared landmark (standard map) ...")
    initC, resC, dC = solve_variant(gi, gi.obs, [], cfg, cfg.ref_id)

    def landmark_gap(markers):
        gaps = [pose_gap(markers[o], markers[c]) for c, o in copies.items() if c in markers and o in markers]
        return (float(np.mean([g[0] for g in gaps])), float(np.mean([g[1] for g in gaps]))) if gaps else (0.0, 0.0)

    rows = []
    for label, cams, markers, drop in (("initial guess (odometry only)", initA[:2], initA[2], 0),
                                       ("A  solved, no loop closure", (resA.cam_R, resA.cam_p), resA.marker_poses, dA),
                                       ("B  solved + explicit loop closure", (resB.cam_R, resB.cam_p), resB.marker_poses, dB),
                                       ("C  shared landmark (implicit LC)", (resC.cam_R, resC.cam_p), resC.marker_poses, dC)):
        te, re = rel_edge_residuals(cams[0], cams[1], edges)
        lg = landmark_gap(markers)
        rec = {"label": label, "edge_t": float(te.mean()), "edge_r": float(re.mean()), "lm_t": lg[0], "lm_r": lg[1]}
        if not label.startswith("initial"):
            ref_inv = inv_T(markers[cfg.ref_id])
            poses = {m: ref_inv @ T for m, T in markers.items() if m in gt}      # original ids only
            mm = MarkerMap(cfg.ref_id, poses)
            t = error_table(mm, gt)
            rec.update(rms_map=t["rms_map"], rms_held=t["rms_held"], pair_rms=t["pair_rms"], scale=t["scale"],
                       T_rw=fit_room_frame(mm, gt)[0], ref_inv=ref_inv, cam_p=cams[1], poses=poses)
        rows.append(rec)

    print("\nloop-closure residual = gap between the two estimates of the same place")
    print(f"{'stage':<36} {'loop-edge gap':>16} {'landmark-copy gap':>20} {'map RMS 0-6':>12} {'held-out':>9} {'pairs RMS':>10}")
    md = ["| stage | loop-edge gap (m / deg) | landmark-copy gap (m / deg) | map RMS markers 0-6 (cm) | held-out 7-9 (cm) | pairwise RMS (cm) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        em = lambda k: f"{100 * r[k]:.1f}" if k in r else "-"
        lm = f"{r['lm_t']:.2f} / {r['lm_r']:.1f}" if copies else "-"
        print(f"{r['label']:<36} {r['edge_t']:>8.2f} / {r['edge_r']:>5.1f} {lm:>20} {em('rms_map'):>12} {em('rms_held'):>9} {em('pair_rms'):>10}")
        md.append(f"| {r['label']} | {r['edge_t']:.2f} / {r['edge_r']:.1f} | {lm} | {em('rms_map')} | {em('rms_held')} | {em('pair_rms')} |")
    print(f"(outlier observations dropped: A {dA}, B {dB}, C {dC})")
    (out / f"loop_closure_{name}.md").write_text("\n".join(md) + "\n")

    fig, ax = plt.subplots(figsize=(8, 8))
    for r, col, lab in ((rows[1], "tab:red", "A no loop closure"), (rows[2], "tab:green", "B + loop closure"), (rows[3], "0.4", "C shared landmark")):
        p = (r["T_rw"][:3, :3] @ (r["ref_inv"][:3, :3] @ r["cam_p"].T + r["ref_inv"][:3, 3:4])).T + r["T_rw"][:3, 3]
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.4, label=lab)
    for m, pos in gt.items():
        ax.plot(*pos["pos"][:2], "x", color="k", ms=6)
        ax.annotate(str(m), pos["pos"][:2], textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(fontsize=8)
    ax.set_title(f"{name}: camera path with / without loop closure (x = surveyed markers)")
    fig.tight_layout()
    fig.savefig(out / f"loop_closure_{name}.png", dpi=140)
    print(f"\nsaved {out / f'loop_closure_{name}.md'}\nsaved {out / f'loop_closure_{name}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
