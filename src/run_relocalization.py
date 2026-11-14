"""Cold-start relocalization of a run against a saved marker map that was built WITHOUT that run.

    python -m src.run_relocalization "data/all runs/run 4" --map outputs/marker_map_excl_run4.json

Replay starts at frame 0 with no pose estimate, no odometry and no prior. Each frame's accepted marker
detections (detector gate: pixel area, IPPE ambiguity; ids restricted to the map's in_map landmarks) each give
a camera pose in the room frame: T_room_cam = T_room_marker @ inv(T_cm). A FIX is accepted when
  - two or more markers in one frame give agreeing poses (<= 15 cm and <= 10 deg), or
  - one marker gives `--stable-frames` consecutive frames whose positions each move <= 5 cm.
Relocalization time = seconds from the first replayed frame to the first accepted fix.
Relocalization error = distance between the fix position and the position obtained from the SURVEYED
marker pose (data/markers.csv; wall markers upright) with the same observation, i.e. how far the mapped
marker pose being used is from the survey, seen from the camera. (No independent ground-truth camera
position exists at that instant: room.md only gives the start tape and the run-1 path.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_run import detect_run
from src.geometry import inv_T, make_T, rot_log, T_from_rvec_tvec
from src.mapping.marker_map import MapConfig, MarkerMap, _gate, load_ground_truth


def surveyed_marker_pose(gt_entry, upside_down: bool = True):
    """4x4 room pose of a wall marker from the survey: z = outward normal; in-plane y = world -Z.

    markers.csv gives position and facing only. The in-plane rotation is NOT surveyed; every mapped wall marker
    (both maps, 7 markers) has its y axis within 2-8 deg of world DOWN, so the markers are mounted rotated
    180 deg in-plane and the truth uses that (upside_down=False would assume upright, which is wrong here).
    None for floor markers.
    """
    n = gt_entry["normal"]
    if abs(n[2]) > 0.5:
        return None
    y = np.array([0, 0, -1.0 if upside_down else 1.0])
    x = np.cross(y, n)
    x /= np.linalg.norm(x)
    return make_T(np.column_stack([x, np.cross(n, x), n]), gt_entry["pos"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--map", required=True, help="marker map built without this run")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--stable-frames", type=int, default=5)
    ap.add_argument("--use-all-markers", action="store_true", help="also use the floor scoring markers as landmarks")
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    name = Path(a.run_dir).resolve().name.strip().replace(" ", "_")
    ds = RunDataset(a.run_dir)
    det = _gate(detect_run(ds, cache_path=out / f"{name}_detections.npz"), MapConfig())
    mmap = MarkerMap.load(a.map)
    gt = load_ground_truth()
    ids = set(mmap.poses) if a.use_all_markers else {m for m, f in mmap.in_map.items() if f}
    markers = {m: mmap.T_room_world @ mmap.poses[m] for m in ids}
    src_run = mmap.info.get("source_run", "?")
    print(f"map {a.map}: {len(mmap.poses)} markers placed, landmarks used {sorted(ids)}; "
          f"{name} was not used to build it (info: runs pooled in the map file were chosen by the caller)")

    t0 = int(ds.frame_t_ns[0])
    by_frame: dict[int, dict] = {}
    for f, m, rv, tv in zip(det.frame, det.marker_id, det.rvec, det.tvec):
        if int(m) in markers:
            by_frame.setdefault(int(f), {})[int(m)] = T_from_rvec_tvec(rv, tv)

    fixes = []                 # (frame, position_map, position_survey or None, markers used)
    streak = {}                # marker -> (last frame, last position, count)
    first = None
    for f in sorted(by_frame):
        cand = {m: markers[m] @ inv_T(T) for m, T in by_frame[f].items()}
        accepted_markers = []
        ms = sorted(cand)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                A, B = cand[ms[i]], cand[ms[j]]
                if np.linalg.norm(A[:3, 3] - B[:3, 3]) <= 0.15 and np.degrees(np.linalg.norm(rot_log(A[:3, :3].T @ B[:3, :3]))) <= 10:
                    accepted_markers = ms
        new_streak = {}
        for m, T in cand.items():
            last = streak.get(m)
            if last and f - last[0] <= 3 and np.linalg.norm(T[:3, 3] - last[1]) <= 0.05:
                new_streak[m] = (f, T[:3, 3], last[2] + 1)
            else:
                new_streak[m] = (f, T[:3, 3], 1)
            if new_streak[m][2] >= a.stable_frames and not accepted_markers:
                accepted_markers = [m]
        streak = new_streak
        if accepted_markers:
            pos = np.mean([cand[m][:3, 3] for m in accepted_markers], axis=0)
            truth = [surveyed_marker_pose(gt[m]) @ inv_T(by_frame[f][m]) for m in accepted_markers if surveyed_marker_pose(gt[m]) is not None]
            fixes.append((f, pos, np.mean([T[:3, 3] for T in truth], axis=0) if truth else None, accepted_markers))
            if first is None:
                first = fixes[-1]

    if first is None:
        print("no accepted fix in the whole run")
        return 1
    f, pos, truth, used = first
    t_fix = (int(ds.frame_t_ns[f]) - t0) * 1e-9
    raw_first = int(det.frame.min())
    print(f"\nfirst raw gated detection of a landmark: frame {min(by_frame)} (t={(int(ds.frame_t_ns[min(by_frame)]) - t0) * 1e-9:.2f} s)")
    print(f"first ACCEPTED fix: frame {f}, markers {used}, t = {t_fix:.2f} s after the start of replay")
    print(f"fix position in room frame: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
    if truth is not None:
        err = float(np.linalg.norm(pos - truth))
        print(f"relocalization error vs surveyed-marker pose: {100 * err:.1f} cm (xy {100 * np.linalg.norm((pos - truth)[:2]):.1f} cm)")
    else:
        err = float("nan")
        print("relocalization error: n/a (fix used only floor markers, no surveyed orientation)")

    errs = np.array([np.linalg.norm(p - t) for _, p, t, _ in fixes if t is not None])
    n_fix_frames = len(fixes)
    print(f"\nafter the first fix: {n_fix_frames} accepted-fix frames out of {len(by_frame)} frames that saw a landmark "
          f"({len(ds.frame_t_ns)} frames total); error vs survey over all fixes: median {100 * np.median(errs):.1f} cm, "
          f"90% {100 * np.percentile(errs, 90):.1f} cm, max {100 * errs.max():.1f} cm")
    md = [f"| run | map | landmarks | first accepted fix (s) | fix markers | relocalization error (cm) | median error over all fixes (cm) |",
          "|---|---|---|---|---|---|---|",
          f"| {name} | {Path(a.map).name} | {sorted(ids)} | {t_fix:.2f} | {used} | {100 * err:.1f} | {100 * np.median(errs):.1f} |"]
    (out / f"relocalization_{name}.md").write_text("\n".join(md) + "\n")
    print(f"saved {out / f'relocalization_{name}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
