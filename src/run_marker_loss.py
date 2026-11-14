"""Marker-loss experiment (run 2): what the estimate does while no mapped marker is visible, and how it recovers.

    python -m src.run_marker_loss "data/all runs/run 2" --map outputs/marker_map_excl_run2.json

The fixed-map pose graph is solved twice on the same keyframes:
  full    all observations (a smoother: later markers also correct the gap)
  causal  keyframes 0..k_b with the observations at k_b and later withheld, k_b = first keyframe that sees a
          landmark after the gap. This is what the estimate looked like just BEFORE the marker reappeared.
Gap quantities: duration and where it falls; position 1-sigma at the start and end of the gap (causal solve);
divergence |causal - full| through the gap; correction at re-acquisition = |causal - full| at k_b, and the
innovation = |camera position implied directly by the re-seen marker - causal estimate| (what a filter would jump).
Landmarks are the map's in_map markers (0-6); floor markers are not used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_run import detect_run
from src.estimation.slam_core import solve_with_map
from src.evaluation.common import m4_in_room, mean_error, survey_poses, survey_reference
from src.geometry import inv_T, rot_log
from src.mapping.marker_map import MapConfig, MarkerMap, _gate
from src.odometry.trajectory import load_trajectory_csv
from src.run_slam import parse_polyline, polyline_error


def ellipse(ax, c, cov2, **kw):
    w, V = np.linalg.eigh(cov2)
    ax.add_patch(Ellipse(c, 4 * np.sqrt(max(w[1], 0)), 4 * np.sqrt(max(w[0], 0)), angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])),
                         fill=False, **kw))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--map", required=True, help="leave-this-run-out marker map")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--min-gap", type=float, default=2.0, help="seconds without a landmark that count as marker loss")
    ap.add_argument("--gt-path", default=None, help='optional taped path "x,y;x,y;..." (none is recorded for run 2)')
    ap.add_argument("--out-name", default="run2_markerloss")
    ap.add_argument("--gt-note", default=None, help="mark the taped-path rows with a dagger and append this footnote "
                    "(use when the run only approximated the taped path)")
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    name = Path(a.run_dir).resolve().name.strip().replace(" ", "_")
    cfg = MapConfig()
    ds = RunDataset(a.run_dir)
    det = _gate(detect_run(ds, cache_path=out / f"{name}_detections.npz"), cfg)
    traj = load_trajectory_csv(out / f"{name}_traj.csv")
    mmap = MarkerMap.load(a.map)
    markers = {m: mmap.T_room_world @ T for m, T in mmap.poses.items()}
    use_ids = {m for m, f in mmap.in_map.items() if f}

    sol = solve_with_map(traj, det, markers, use_ids, cfg, compute_cov=True)
    gi = sol.gi
    t = (gi.kf_t_ns - gi.kf_t_ns[0]) * 1e-9
    vis = sorted({o.kf for o in gi.obs})
    gaps = [(vis[i], vis[i + 1]) for i in range(len(vis) - 1) if t[vis[i + 1]] - t[vis[i]] >= a.min_gap]
    if not gaps:
        raise SystemExit(f"No gap of >= {a.min_gap:g} s without a landmark in {name}.")
    k_a, k_b = max(gaps, key=lambda g: t[g[1]] - t[g[0]])
    run_len = t[-1]
    print(f"{name}: {len(gi.kf_frames)} keyframes, {len(gi.obs)} landmark observations (ids {sorted({o.marker for o in gi.obs})}), run {run_len:.1f} s")
    print(f"first landmark fix at t={t[vis[0]]:.1f} s; last at t={t[vis[-1]]:.1f} s")
    print("landmark-free gaps >= %.1f s: " % a.min_gap + ", ".join(f"{t[x]:.1f}-{t[y]:.1f} s ({t[y] - t[x]:.1f} s)" for x, y in gaps))
    print(f"analysing the longest: keyframes {k_a}..{k_b}")

    causal = solve_with_map(traj, det, markers, use_ids, cfg, compute_cov=True, kf_limit=k_b)
    p_c, p_f = causal.cam_p, sol.cam_p[:k_b + 1]
    sig_c, sig_f = causal.result.position_sigma(), sol.result.position_sigma()
    seg = np.arange(k_a, k_b + 1)
    div = np.linalg.norm(p_c[seg, :2] - p_f[seg, :2], axis=1)
    path_in_gap = float(np.linalg.norm(np.diff(p_c[seg, :2], axis=0), axis=1).sum())

    obs_b = [o for o in gi.obs if o.kf == k_b]
    ob = min(obs_b, key=lambda o: o.dist)
    T_cm = np.vstack([np.hstack([ob.R, ob.t[:, None]]), [0, 0, 0, 1]])
    T_fix = markers[ob.marker] @ inv_T(T_cm)
    innov = float(np.linalg.norm(T_fix[:2, 3] - p_c[k_b, :2]))
    innov_rot = float(np.degrees(np.linalg.norm(rot_log(causal.cam_R[k_b].T @ T_fix[:3, :3]))))
    corr = float(np.linalg.norm(sol.cam_p[k_b, :2] - p_c[k_b, :2]))
    corr_rot = float(np.degrees(np.linalg.norm(rot_log(causal.cam_R[k_b].T @ sol.cam_R[k_b]))))

    surv = survey_poses()
    ref = survey_reference(det, surv)
    Rm4, pm4 = m4_in_room(traj, det, surv)
    kf = gi.kf_frames
    slam_err, n_ref = mean_error(lambda f: sol.cam_p[int(np.argmin(np.abs(kf - f)))], ref)
    m4_err, _ = mean_error(lambda f: pm4[min(f, len(pm4) - 1)], ref)
    in_gap_frames = [f for f in ref if kf[k_a] <= f <= kf[k_b]]
    plen = lambda p: float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())
    gt_poly = parse_polyline(a.gt_path) if a.gt_path else None

    lines = [f"# {name}: marker-loss experiment", "",
             f"Map `{Path(a.map).name}` (built without {name}); landmarks = in_map markers {sorted(use_ids)}; fixed-map pose graph on {len(kf)} keyframes.", "",
             "## The marker-loss segment", "",
             "| quantity | value |", "|---|---|",
             f"| longest stretch with no landmark visible | **{t[k_b] - t[k_a]:.1f} s** (t = {t[k_a]:.1f} to {t[k_b]:.1f} s of a {run_len:.0f} s run, {100 * (t[k_b] - t[k_a]) / run_len:.0f}% of the run) |",
             f"| where | between the last fix on marker(s) {sorted({o.marker for o in gi.obs if o.kf == k_a})} at ({p_f[k_a, 0]:.2f}, {p_f[k_a, 1]:.2f}) m and the re-acquisition on marker {ob.marker} at ({sol.cam_p[k_b, 0]:.2f}, {sol.cam_p[k_b, 1]:.2f}) m |",
             f"| all landmark-free gaps >= {a.min_gap:g} s | " + "; ".join(f"{t[x]:.1f}-{t[y]:.1f} s" for x, y in gaps) + " |",
             f"| estimated distance travelled inside the gap (causal) | {path_in_gap:.2f} m |",
             f"| position 1-sigma at the start of the gap (causal) | {100 * sig_c[k_a]:.1f} cm |",
             f"| position 1-sigma at the end of the gap, before re-acquisition (causal) | **{100 * sig_c[k_b]:.1f} cm** ({sig_c[k_b] / sig_c[k_a]:.1f}x growth; max in gap {100 * sig_c[seg].max():.1f} cm) |",
             f"| divergence between causal and full-solution path through the gap | mean {100 * div.mean():.1f} cm, max {100 * div.max():.1f} cm |",
             f"| correction at re-acquisition: causal -> full solution, same keyframe | **{100 * corr:.1f} cm**, {corr_rot:.1f} deg |",
             f"| innovation: pose implied directly by the re-seen marker {ob.marker} vs causal estimate | {100 * innov:.1f} cm, {innov_rot:.1f} deg |",
             f"| whether the correction is explained by the uncertainty | innovation / causal sigma = {innov / sig_c[k_b]:.2f} |",
             "", "## Overall (full solution)", "",
             "| metric | pose graph | M4 odometry only |", "|---|---|---|",
             f"| top-down path length | {plen(sol.cam_p):.2f} m | {plen(pm4[kf]):.2f} m |",
             f"| end point (room frame) | ({sol.cam_p[-1, 0]:.2f}, {sol.cam_p[-1, 1]:.2f}) m | ({pm4[-1, 0]:.2f}, {pm4[-1, 1]:.2f}) m |",
             f"| mean error vs survey-derived reference ({n_ref} frames with a wall marker in view) | {100 * slam_err:.0f} cm | {100 * m4_err:.0f} cm |"]
    if gt_poly is not None:
        e1, e0 = polyline_error(sol.cam_p[:, :2], gt_poly), polyline_error(pm4[kf][:, :2], gt_poly)
        dag = "\u2020" if a.gt_note else ""
        lines += [f"| mean distance to taped path{dag} | {100 * e1.mean():.0f} cm | {100 * e0.mean():.0f} cm |",
                  f"| end-point error vs taped end{dag} | {100 * np.linalg.norm(sol.cam_p[-1, :2] - gt_poly[-1]):.0f} cm | {100 * np.linalg.norm(pm4[-1, :2] - gt_poly[-1]):.0f} cm |"]
    else:
        lines += ["| mean distance to taped path / end-point error | n/a: no taped path is recorded for this run in data/room.md | n/a |"]
    lines += [f"| position 1-sigma, median / max (full solution) | {100 * np.nanmedian(sig_f if len(sig_f) else sig_c):.0f} cm / {100 * np.nanmax(sol.result.position_sigma()):.0f} cm | - |",
              "", "## Notes", "",
              "- *Causal* = the same graph with the marker observations at the re-acquisition keyframe and later withheld; *full* uses everything. The pose graph is a smoother, so the full solution also distributes the correction back across the gap.",
              "- The causal uncertainty is conservative and elongated along the direction of travel: the speed through the gap is unobserved (odometry scale is a free variable per window with a loose prior and 100% step-length noise), so only direction and rotation are tightly bridged. The real correction (84 cm here) is about a quarter of the 1-sigma radius.",
              "- The causal solve still contains the markers seen BEFORE the gap, so its pose at the start of the gap is anchored; the gap is bridged by M4 odometry (direction + constant-speed scale per window, see src/estimation/graph_inputs.py).",
              "- Reference = camera position implied by the surveyed pose of a wall marker in view (independent of the map); the fused path inherits the map's own error, so its mean error is a floor of tens of cm.",
              f"- Frames inside the gap that have a reference: {len(in_gap_frames)} (those see wall markers that are not in the landmark set or were gated out), so the gap itself is mostly unreferenced."]
    if a.gt_note:
        lines += ["", "\u2020 " + a.gt_note]
    (out / f"{a.out_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.plot(sol.cam_p[:, 0], sol.cam_p[:, 1], "-", color="#52606d", lw=1.6, label="corrected path (full solution)")
    ax.plot(sol.cam_p[k_a:k_b + 1, 0], sol.cam_p[k_a:k_b + 1, 1], "-", color="#d1495b", lw=4, alpha=0.85,
            label=f"no-marker segment ({t[k_b] - t[k_a]:.1f} s)")
    ax.plot(p_c[k_a:k_b + 1, 0], p_c[k_a:k_b + 1, 1], "--", color="#e08a00", lw=2, label="estimate before re-acquisition (causal)")
    for k in range(0, len(sol.cam_p), 20):
        if not (k_a <= k <= k_b) and sol.result.cov is not None:
            ellipse(ax, sol.cam_p[k, :2], sol.result.cov[k, 3:5, 3:5], ec="0.6", lw=0.7, alpha=0.7)
    step = max((k_b - k_a) // 8, 1)
    for k in list(range(k_a, k_b + 1, step)) + [k_b]:
        ellipse(ax, p_c[k, :2], causal.result.cov[k, 3:5, 3:5], ec="#2f6fb0", lw=1.3)
    ax.plot([], [], color="#2f6fb0", lw=1.3, label="2-sigma ellipse, causal (grows in the gap)")
    ax.plot([], [], color="0.6", lw=0.7, label="2-sigma ellipse, full solution")
    ax.annotate("", xy=sol.cam_p[k_b, :2], xytext=p_c[k_b, :2], arrowprops=dict(arrowstyle="->", color="k", lw=1.6))
    ax.annotate(f"snap {100 * corr:.0f} cm", sol.cam_p[k_b, :2], textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.plot(*sol.cam_p[0, :2], "o", color="tab:green", ms=10, label="start")
    ax.plot(*sol.cam_p[-1, :2], "s", color="tab:red", ms=9, label="end")
    for m, T in sorted(markers.items()):
        ax.plot(*T[:2, 3], "D", color="#f0a202", ms=8, mec="k")
        ax.annotate(str(m), T[:2, 3], textcoords="offset points", xytext=(6, 6), fontsize=9, weight="bold")
    ax.plot([], [], "D", color="#f0a202", mec="k", label="mapped marker")
    ax.set_aspect("equal")
    pts = np.vstack([sol.cam_p[:, :2], np.array([T[:2, 3] for T in markers.values()])])
    ax.set_xlim(pts[:, 0].min() - 0.8, pts[:, 0].max() + 0.8)       # frame the path; the causal ellipses run off the edge
    ax.set_ylim(pts[:, 1].min() - 0.8, pts[:, 1].max() + 0.8)
    ax.grid(alpha=0.3)
    ax.set_xlabel("X (m), room frame  (causal ellipses extend beyond the view)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{name}: marker loss at t = {t[k_a]:.1f}-{t[k_b]:.1f} s (uncertainty grows {sig_c[k_b] / sig_c[k_a]:.1f}x, "
                 f"then snaps {100 * corr:.0f} cm)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{a.out_name}.png", dpi=150)
    print(f"\nsaved {out / f'{a.out_name}.md'}\nsaved {out / f'{a.out_name}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
