"""Marker-density ablation: endpoint drift vs number of markers whose detections are used.

    python -m src.run_ablation "data/all runs/run 3"

Run 3 is the only run that sees all 10 markers. Endpoint drift = distance (top-down) from the estimated
final camera position to a reference built from the SURVEYED pose of a wall marker visible in the last
frames (so it does not depend on the map or on which markers the solve used). For each marker count (10, 8, 6, 4, 2, 0) several random ID subsets are
kept (all others' detections ignored), the fixed-map pose graph is solved, and the spread over subsets is
shown. 0 markers = the M4 visual-inertial trajectory with no marker input, placed in the room frame by the same first-fix anchor as the baseline table.
Two maps: `pooled` (built from runs 1-5, i.e. including this run: optimistic, landmarks agree with this
run's own sightings) and `leave-out` (built without this run: honest, but that map is itself poor, ~90 cm).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COUNTS = [10, 8, 6, 4, 2, 0]


def _task(args):
    run_name, out_dir, map_path, ids = args
    from src.detection.detect_markers import load_calibration
    from src.detection.detect_run import RunDetections
    from src.estimation.slam_core import solve_with_map
    from src.geometry import T_from_rvec_tvec, inv_T
    from src.mapping.marker_map import MapConfig, MarkerMap, _gate, load_ground_truth
    from src.odometry.trajectory import load_trajectory_csv
    from src.run_baselines import anchor_poses, register
    from src.run_relocalization import surveyed_marker_pose
    out = Path(out_dir)
    cfg = MapConfig()
    gt = load_ground_truth()
    surv = {m: surveyed_marker_pose(gt[m]) for m in gt if surveyed_marker_pose(gt[m]) is not None}    # wall markers
    det = _gate(RunDetections.load(out / f"{run_name}_detections.npz"), cfg)
    traj = load_trajectory_csv(out / f"{run_name}_traj.csv")
    # per-frame reference positions from the SURVEYED pose of whichever wall markers are visible (independent
    # of the map and of which markers the solve used)
    ref = {}
    for f, m, rv, tv in zip(det.frame, det.marker_id, det.rvec, det.tvec):
        if int(m) in surv:
            ref.setdefault(int(f), []).append((surv[int(m)] @ inv_T(T_from_rvec_tvec(rv, tv)))[:3, 3])
    ref = {f: np.mean(v, axis=0) for f, v in ref.items()}
    frames_ref = np.array(sorted(ref))
    f_last = int(frames_ref[-1])
    p_end_ref = np.mean([ref[f] for f in frames_ref if f >= f_last - 5], axis=0)
    sol = None
    if ids:
        mm = MarkerMap.load(map_path)
        markers = {m: mm.T_room_world @ T for m, T in mm.poses.items()}
        sol = solve_with_map(traj, det, markers, set(ids), cfg)
    if sol is None:                                   # no marker input: M4 registered by the same first-fix anchor
        f0, anchor = anchor_poses(det, surv, set(surv))
        _, pfull = register(traj.R_wc, traj.p, anchor)
        at = lambda f: pfull[min(f, len(pfull) - 1)]
        nobs = 0
    else:
        kf = sol.gi.kf_frames
        at = lambda f: sol.cam_p[int(np.argmin(np.abs(kf - f)))]
        nobs = len(sol.gi.obs)
    err = np.array([np.linalg.norm(at(f)[:2] - ref[f][:2]) for f in frames_ref[::3]])
    return float(np.linalg.norm(at(f_last)[:2] - p_end_ref[:2])), float(err.mean()), nobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--pooled-map", default="outputs/marker_map.json")
    ap.add_argument("--leaveout-map", default="outputs/marker_map_excl_run3.json")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--subsets", type=int, default=6, help="random ID subsets per marker count")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    name = Path(a.run_dir).resolve().name.strip().replace(" ", "_")
    out = Path(a.out_dir)
    rng = np.random.default_rng(a.seed)
    subsets = {}
    for c in COUNTS:
        if c in (10, 0):
            subsets[c] = [tuple(range(10))[:c]] if c else [()]
        else:
            seen, lst = set(), []
            while len(lst) < a.subsets:
                s = tuple(sorted(rng.choice(10, c, replace=False).tolist()))
                if s not in seen:
                    seen.add(s)
                    lst.append(s)
            subsets[c] = lst

    maps = {"pooled map (includes this run, optimistic)": a.pooled_map, "leave-this-run-out map (honest)": a.leaveout_map}
    jobs, keys = [], []
    for label, mp in maps.items():
        for c in COUNTS:
            for s in subsets[c]:
                jobs.append((name, str(out), mp, s))
                keys.append((label, c, s))
    print(f"{len(jobs)} solves over {a.workers} workers ...")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(_task, jobs))

    res = {label: {c: [] for c in COUNTS} for label in maps}
    for (label, c, sub_ids), (end, ate, nobs) in zip(keys, results):
        res[label][c].append((end, ate, sub_ids, nobs))

    def stat(l, c, i):
        v = np.array([x[i] for x in res[l][c]])
        return v.mean(), v.min(), v.max()

    md = ["| markers used | " + " | ".join(f"{l}: endpoint / mean error, m" for l in maps) + " |", "|---|" + "---|" * len(maps)]
    for c in COUNTS:
        cells = []
        for l in maps:
            e, t = stat(l, c, 0), stat(l, c, 1)
            cells.append(f"{e[0]:.2f} ({e[1]:.2f}-{e[2]:.2f}) / {t[0]:.2f} ({t[1]:.2f}-{t[2]:.2f}), n={len(res[l][c])}")
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    md += ["", f"endpoint = top-down distance from the final estimated camera position of {name} to the position implied by the surveyed pose of a wall marker seen in the last frames.",
           "mean error = same comparison averaged over every 3rd frame in which a wall marker is visible (survey-derived reference, independent of the map and of which markers were used).",
           "0 markers = M4 visual-inertial trajectory registered at its first marker fix. Its endpoint is deceptively small: M4's path is badly shrunk, so a lap ends near its start; the mean error shows the real failure.",
           "'pooled' map was built with this run's own detections (optimistic); 'leave-out' map without them (its own error is ~90 cm RMS). Both maps carry their own error into every row, so rows 10..4 differ little.",
           "others = fixed-map pose graph using only the listed IDs' detections; spread is over random ID subsets."]
    (out / "ablation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    ink, sub, grid = "#1f2933", "#52606d", "#e4e7eb"
    cols = {"pooled map (includes this run, optimistic)": "#2f6fb0", "leave-this-run-out map (honest)": "#c8741a"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    for ax, i, ttl in ((axes[0], 0, "Endpoint drift (m)"), (axes[1], 1, "Mean error over the run (m)")):
        for l, col in cols.items():
            xs = np.array(COUNTS)
            mean = np.array([stat(l, c, i)[0] for c in COUNTS])
            lo = np.array([stat(l, c, i)[1] for c in COUNTS])
            hi = np.array([stat(l, c, i)[2] for c in COUNTS])
            for c in COUNTS:
                ax.plot([c] * len(res[l][c]), [x[i] for x in res[l][c]], "o", color=col, alpha=0.25, ms=5, mec="none")
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(xs, mean, "-o", color=col, lw=2, ms=7, mec="white", mew=1.5, label=l)
        ax.set_title(ttl, fontsize=11, color=ink, loc="left")
        ax.set_xlabel("number of markers whose detections are used", color=sub)
        ax.set_xticks(COUNTS)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=sub)
    axes[0].invert_xaxis()
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"Marker-density ablation on {name} (dots = ID subsets, band = min-max, line = mean). "
                 "0 markers = M4 only (its endpoint is small only because its path is shrunk)", fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=150)
    print(f"saved {out / 'ablation.md'}\nsaved {out / 'ablation.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
