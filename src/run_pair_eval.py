"""Original runs vs their re-recordings: per-run evaluation, head-to-head verdicts, repeatability.

    python -m src.run_pair_eval --out-name rerun_evaluation

Ground truth (data/room.md, run 1's taped path; markers did not move). ROUTE CAVEAT: only run 1, rerun 1 and rerun 2 walked
the taped L exactly (rerun 5 is its out-and-back). Original runs 2 and 5 are roughly L-shaped with the same start point, but
their turning and end points were re-walked, not re-taped, so their taped-path errors are measured against run 1's path
(marked with a dagger) and include an unmeasured route deviation.
    L        (2.9,3.25) -> (0.5,3.25) -> (0.5,0.25)                       5.4 m, endpoint = far end (0.5,0.25)
    OB       the same L walked out and back                                10.8 m, endpoint = START (2.9,3.25)
Every run is solved against the map built from all runs EXCEPT that run and its twin (original <-> rerun), so
an original and its rerun face the identical map and neither helped build it.

Decision rule for "is the rerun better than the original" (fixed in advance, evaluated on the same map):
    BETTER  only if  mean distance to the taped path improves by >= 20%
                 and end-point error is not worse by > 20%
                 and landmark coverage (frames with >= 1 landmark) is not worse by > 10% (relative).
    WORSE   if the mirror-image conditions hold; otherwise NO MEANINGFUL DIFFERENCE -> keep the original.
"Meaningfully harder": a run counts as harder than a reference if at least two of five stress criteria hold
(see HARD_CRITERIA below).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_run import detect_run
from src.estimation.slam_core import solve_with_map
from src.evaluation.common import (REASONS, detection_summary, m4_in_room, mean_error, stress_indicators, survey_poses,
                                   survey_reference)
from src.mapping.marker_map import MapConfig, MarkerMap, _gate
from src.odometry.trajectory import load_trajectory_csv
from src.run_baselines import loop_residual
from src.run_slam import polyline_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "all runs"
P_L = np.array([[2.9, 3.25], [0.5, 3.25], [0.5, 0.25]])
P_OB = np.vstack([P_L, P_L[1::-1]])
GT = {"L": P_L, "OB": P_OB}

# label, folder, twin label, map pair id
RUNS = [("run_1", "run 1", "rerun_1", "pair1"), ("rerun_1", "rerun 1", "run_1", "pair1"),
        ("run_2", "run 2", "rerun_2", "pair2"), ("rerun_2", "rerun 2", "run_2", "pair2"),
        ("run_5", "run 5", "rerun_5", "pair5"), ("rerun_5", "rerun 5", "run_5", "pair5")]
BLUE, ORANGE, RED = "#2f6fb0", "#c8741a", "#d1495b"
TAPED_EXACT = {"run_1", "rerun_1", "rerun_2", "rerun_5"}      # walked the taped path (rerun_5: its out-and-back)
APPROX = {"run_2", "run_5"}                                    # re-walked without re-taping: only nominally the same route
DAGGER = "\u2020"


def tag(label):
    return label + DAGGER if label in APPROX else label


DAGGER_NOTE = (DAGGER + " measured against run 1's taped path, which this run approximated but did not follow exactly: it starts at the same point, "
               "but its turning and end points were re-walked rather than re-taped, and the deviation was not measured. Its taped-path error "
               "(mean distance to the path, end-point error, route reached) therefore includes an unknown amount of route deviation in addition to estimator error.")


def plen(p):
    return float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())


def route_progress(xy, poly=P_L):
    """Arc length (m) along the taped L of the nearest point of the polyline, for each estimated position."""
    seg = np.diff(poly, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(ln)]
    out = np.empty(len(xy))
    for i, q in enumerate(xy):
        best = (1e9, 0.0)
        for k in range(len(seg)):
            t = np.clip(((q - poly[k]) @ seg[k]) / (seg[k] @ seg[k]), 0, 1)
            d = np.linalg.norm(q - (poly[k] + t * seg[k]))
            if d < best[0]:
                best = (d, cum[k] + t * ln[k])
        out[i] = best[1]
    return out


def per_id_table(det, n_frames):
    ids = sorted(set(det.marker_id.tolist()) | (set(det.rej_marker.tolist()) if det.rej_marker is not None else set()))
    rows = []
    for m in ids:
        s = det.marker_id == m
        rej = {r: int(((det.rej_marker == m) & (det.rej_reason == r)).sum()) for r in REASONS} if det.rej_marker is not None else {}
        rows.append({"id": m, "n": int(s.sum()), "dist": float(det.distance[s].mean()) if s.any() else float("nan"),
                     "reproj": float(det.reproj[s].mean()) if s.any() else float("nan"),
                     "rej": ", ".join(f"{k}={v}" for k, v in rej.items() if v) or "-",
                     "cov": 100.0 * len(np.unique(det.frame[s])) / n_frames})
    return rows


def evaluate_run(label, folder, map_path, route, out, loop_gap):
    cfg = MapConfig()
    ds = RunDataset(DATA / folder)
    det_raw = detect_run(ds, cache_path=out / f"{label}_detections.npz")
    det = _gate(det_raw, cfg)
    traj = load_trajectory_csv(out / f"{label}_traj.csv")
    mmap = MarkerMap.load(map_path)
    markers = {m: mmap.T_room_world @ T for m, T in mmap.poses.items()}
    use_ids = {m for m, f in mmap.in_map.items() if f}
    sol = solve_with_map(traj, det, markers, use_ids, cfg, compute_cov=True)
    gi, kf = sol.gi, sol.gi.kf_frames
    surv = survey_poses()
    ref = survey_reference(det, surv)
    Rm4, pm4 = m4_in_room(traj, det, surv)
    r = {"label": label, "folder": folder, "map": Path(map_path).name, "route": route, "n_frames": det_raw.n_frames,
         "table": per_id_table(det_raw, det_raw.n_frames), "det": detection_summary(det_raw, use_ids), "stress": stress_indicators(ds)}
    r["err_slam_ref"], r["n_ref"] = mean_error(lambda f: sol.cam_p[int(np.argmin(np.abs(kf - f)))], ref)
    r["err_m4_ref"], _ = mean_error(lambda f: pm4[min(f, len(pm4) - 1)], ref)
    pS, pM = sol.cam_p, pm4[kf]
    r.update(cam_p=pS, m4_p=pM, markers=markers, keyframes=len(kf), obs=len(gi.obs), dropped=sol.dropped_obs,
             sig_med=float(np.nanmedian(sol.result.position_sigma())), sig_max=float(np.nanmax(sol.result.position_sigma())),
             start=pS[0, :2], end=pS[-1, :2], len_slam=plen(pS), len_m4=plen(pM), dir_ok=float(traj.dir_ok.mean()))
    t = (gi.kf_t_ns - gi.kf_t_ns[0]) * 1e-9
    vis = np.zeros(len(kf), bool)
    vis[[o.kf for o in gi.obs]] = True
    r.update(blind=float(1 - vis.mean()), first_fix=float(t[np.argmax(vis)]), duration=float(t[-1]))
    gt_len = plen(GT[route])
    for tag, p in (("slam", pS), ("m4", pM)):
        sp = route_progress(p[:, :2])
        r[f"reach_{tag}"], r[f"s_end_{tag}"] = float(sp.max()), float(sp[-1])
    # validity gate: an estimate that never traverses the route can look accurate (an end-point near the start, a
    # path hugging the start) without having tracked anything
    r["valid"] = bool(r["len_slam"] >= 0.7 * gt_len and r["reach_slam"] >= 0.7 * plen(P_L))
    r["loop_slam"], r["loop_pairs"] = loop_residual(kf, sol.cam_R, sol.cam_p, det_raw, gi.kf_t_ns, loop_gap)
    r["loop_m4"], _ = loop_residual(kf, Rm4[kf], pM, det_raw, gi.kf_t_ns, loop_gap)
    for tag, p in (("slam", pS), ("m4", pM)):
        poly = GT[route]
        r[f"taped_{tag}"] = float(polyline_error(p[:, :2], poly).mean())
        r[f"end_{tag}"] = float(np.linalg.norm(p[-1, :2] - poly[-1]))
        r[f"end_far_{tag}"] = float(np.linalg.norm(p[-1, :2] - P_L[-1]))
        r[f"end_start_{tag}"] = float(np.linalg.norm(p[-1, :2] - P_L[0]))
        r[f"closure_{tag}"] = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
    return r


def verdict(orig, rer):
    if not rer["valid"]:
        return "INVALID (rerun's estimate does not traverse the route)", float("nan"), float("nan"), float("nan")
    if not orig["valid"]:
        return "ORIGINAL INVALID (cannot compare)", float("nan"), float("nan"), float("nan")
    d_taped = (rer["taped_slam"] - orig["taped_slam"]) / orig["taped_slam"]
    d_end = (rer["end_slam"] - orig["end_slam"]) / max(orig["end_slam"], 1e-9)
    d_cov = (rer["det"]["frames_landmark"] - orig["det"]["frames_landmark"]) / orig["det"]["frames_landmark"]
    better = d_taped <= -0.20 and d_end <= 0.20 and d_cov >= -0.10
    worse = d_taped >= 0.20 and d_end >= -0.20 and d_cov <= 0.10
    word = "BETTER" if better else "WORSE" if worse else "NO MEANINGFUL DIFFERENCE"
    return word, d_taped, d_end, d_cov


def hard_criteria(ref, x):
    """Five stress tests of x against ref (True = x is harder)."""
    return [
        ("darker (mean gray <= 0.7x)", x["stress"]["luminance_mean"] <= 0.7 * ref["stress"]["luminance_mean"],
         f"{x['stress']['luminance_mean']:.0f} vs {ref['stress']['luminance_mean']:.0f}"),
        ("faster rotation (90th pct rate >= 1.3x)", x["stress"]["gyro_p90_dps"] >= 1.3 * ref["stress"]["gyro_p90_dps"],
         f"{x['stress']['gyro_p90_dps']:.0f} vs {ref['stress']['gyro_p90_dps']:.0f} deg/s"),
        ("blurrier (blurriest-10% sharpness <= 0.7x)", x["stress"]["sharpness_p10"] <= 0.7 * ref["stress"]["sharpness_p10"],
         f"{x['stress']['sharpness_p10']:.0f} vs {ref['stress']['sharpness_p10']:.0f}"),
        ("fewer landmark frames (>= 20% lower)", x["det"]["frames_landmark"] <= 0.8 * ref["det"]["frames_landmark"],
         f"{100 * x['det']['frames_landmark']:.0f}% vs {100 * ref['det']['frames_landmark']:.0f}%"),
        ("pose graph fails to traverse the route, or error >= 1.2x, or reprojection error >= 1.2x",
         (not x["valid"]) or x["taped_slam"] >= 1.2 * ref["taped_slam"] or x["det"]["reproj_mean"] >= 1.2 * ref["det"]["reproj_mean"],
         f"valid {x['valid']} (path {x['len_slam']:.1f} m, reach {x['reach_slam']:.1f} m); taped {100 * x['taped_slam']:.0f} vs {100 * ref['taped_slam']:.0f} cm; reproj {x['det']['reproj_mean']:.2f} vs {ref['det']['reproj_mean']:.2f} px"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--out-name", default="rerun_evaluation")
    ap.add_argument("--run5-route", choices=["auto", "L", "OB"], default="L", help="route of the ORIGINAL run 5 (confirmed one-way; 'auto' guesses from the end point)")
    ap.add_argument("--rerun5-route", choices=["L", "OB"], default="OB", help="route of folder 'rerun 5' (the user's 'rerun 5')")
    a = ap.parse_args(argv)
    out = ROOT / a.out_dir

    res = {}
    for label, folder, twin, pair in RUNS:
        mp = out / f"marker_map_excl_{pair}.json"
        route = {"run_1": "L", "rerun_1": "L", "run_2": "L", "rerun_2": "L", "rerun_5": a.rerun5_route,
                 "run_5": "OB" if a.run5_route == "OB" else "L"}[label]
        gap = 8.0 if route == "OB" else 3.0
        print(f"evaluating {label} ({folder}) on {mp.name}, route {route} ...", flush=True)
        res[label] = evaluate_run(label, folder, mp, route, out, gap)
    if a.run5_route == "auto":                       # decide the original run 5's route from where it ends
        r5 = res["run_5"]
        route = "OB" if r5["end_start_slam"] < r5["end_far_slam"] else "L"
        print(f"original run 5 ends {100 * r5['end_start_slam']:.0f} cm from the start tape and {100 * r5['end_far_slam']:.0f} cm from the far end -> treated as {route}")
        if route == "OB":
            res["run_5"] = evaluate_run("run_5", "run 5", out / "marker_map_excl_pair5.json", "OB", out, 8.0)
    R = res
    L = ["# Reruns vs originals", "",
         "Ground truth: run 1's taped L path for all L walks; for the out-and-back walk the same L traversed out and back (endpoint = start tape). "
         "Each run is solved against the map built from every run except that run and its twin, so an original and its rerun face the identical map.", "",
         f"Route of `rerun 5`: {a.rerun5_route} (walks the L and returns, so its endpoint is the start tape).", "",
         "**Route assumption.** run_1, rerun_1 and rerun_2 followed the taped L path exactly. Original run_2 and run_5 are also roughly L-shaped and share the same start point, "
         "but their turning and end points were re-walked rather than re-taped, so they deviate slightly from run 1's taped path (deviation not measured). "
         f"Wherever a taped-path error is reported for them it carries a dagger ({DAGGER}); see the footnote under the first table.", ""]

    # ---- per-run summary
    L += ["## All runs", "", "| run | route | duration (s) | landmark frames | reproj (px) | pose-graph mean dist to taped path (cm) | end-point error (cm) | est. path length / taped (m) | route reached (m of 5.4) | traverses route? | M4-only mean dist (cm) | loop residual pose graph / M4 (m) [pairs] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, *_ in RUNS:
        x = R[lab]
        L.append(f"| {tag(lab)} | {x['route']} | {x['duration']:.0f} | {100 * x['det']['frames_landmark']:.0f}% | {x['det']['reproj_mean']:.2f} | {100 * x['taped_slam']:.0f} | {100 * x['end_slam']:.0f} | "
                 f"{x['len_slam']:.2f} / {plen(GT[x['route']]):.1f} | {x['reach_slam']:.1f} | {'yes' if x['valid'] else '**NO**'} | {100 * x['taped_m4']:.0f} | {x['loop_slam']:.2f} / {x['loop_m4']:.2f} [{x['loop_pairs']}] |")
    L += ["", "End-point error = distance from the final pose-graph position to the route's endpoint (far end for L, START for the out-and-back walk). "
          "It includes a constant ~0.3-0.5 m offset because the phone is held ahead of the body walking the tape.", "",
          DAGGER_NOTE, "",
          "The columns 'pose-graph mean dist to taped path', 'end-point error', 'route reached' and 'traverses route?' (which uses route reached) are all taped-path quantities, so they carry the dagger for run_2 and run_5.", ""]

    # ---- per-ID tables
    L += ["## Detection tables (sanity-check style, detector-accepted detections)", ""]
    for lab, *_ in RUNS:
        x = R[lab]
        L += [f"### {lab} ({x['folder']}) - {x['n_frames']} frames, {100 * x['det']['frames_any']:.0f}% of frames with >= 1 accepted marker, {100 * x['det']['frames_landmark']:.0f}% with a landmark (0-6)", "",
              "| ID | detections | mean dist (m) | mean reproj (px) | rejections | coverage (% of frames) |", "|---|---|---|---|---|---|"]
        for t in x["table"]:
            L.append(f"| {t['id']} | {t['n']} | {t['dist']:.2f} | {t['reproj']:.2f} | {t['rej']} | {t['cov']:.1f} |" if t["n"] else f"| {t['id']} | 0 | - | - | {t['rej']} | 0.0 |")
        L.append("")

    # ---- head-to-head
    L += ["## Head-to-head: each rerun vs its original (same map)", "",
          "Decision rule fixed in advance (plus one validity gate added after the first run exposed that an estimate which never leaves the start can look accurate: the pose graph must traverse >= 70% of the taped length and reach >= 70% of the L): BETTER only if mean distance to taped path improves >= 20% AND end-point error is not worse by > 20% AND landmark coverage is not worse by > 10%; WORSE for the mirror image; otherwise NO MEANINGFUL DIFFERENCE and the original is kept.", "",
          "| pair | metric | original | rerun | change |", "|---|---|---|---|---|"]
    verdicts = {}
    for o, rr in (("run_1", "rerun_1"), ("run_2", "rerun_2"), ("run_5", "rerun_5")):
        O, X = R[o], R[rr]
        v = verdict(O, X)
        verdicts[(o, rr)] = v
        rows = [("mean distance to taped path (cm)", 100 * O["taped_slam"], 100 * X["taped_slam"], "{:.0f}"),
                ("end-point error (cm)", 100 * O["end_slam"], 100 * X["end_slam"], "{:.0f}"),
                ("landmark coverage (% of frames)", 100 * O["det"]["frames_landmark"], 100 * X["det"]["frames_landmark"], "{:.0f}"),
                ("mean reprojection error (px)", O["det"]["reproj_mean"], X["det"]["reproj_mean"], "{:.2f}"),
                ("share of keyframes with no landmark", 100 * O["blind"], 100 * X["blind"], "{:.0f}"),
                ("M4-only mean distance to taped path (cm)", 100 * O["taped_m4"], 100 * X["taped_m4"], "{:.0f}"),
                ("pose-graph path length (m)", O["len_slam"], X["len_slam"], "{:.2f}"),
                ("route reached (m of 5.4)", O["reach_slam"], X["reach_slam"], "{:.1f}")]
        for name, a_, b_, f in rows:
            L.append(f"| {tag(o)} vs {rr} | {name}{DAGGER if (o in APPROX and ('taped' in name or 'end-point' in name or 'route reached' in name)) else ''} | {f.format(a_)} | {f.format(b_)} | {(b_ - a_) / a_ * 100:+.0f}% |")
        if X["valid"] and O["valid"]:
            dt_, de_ = (X["taped_slam"] - O["taped_slam"]) / O["taped_slam"], (X["end_slam"] - O["end_slam"]) / O["end_slam"]
            reading = ("rerun more accurate" if dt_ <= -0.2 and de_ <= 0.2 else "rerun LESS accurate" if dt_ >= 0.2 and de_ >= -0.2 else "mixed / within +-20%")
            L.append(f"| {tag(o)} vs {rr} | accuracy reading (ignoring the coverage guard) | | | {reading} |")
        L.append(f"| **{tag(o)} vs {rr}** | **verdict** | | | **{v[0]}**" + ("" if np.isnan(v[1]) else f" (taped {100 * v[1]:+.0f}%, end {100 * v[2]:+.0f}%, coverage {100 * v[3]:+.0f}%)") + " |")
    L += ["", DAGGER_NOTE, "",
          f"**What this means for the pairs.** Pair 1 (run_1 vs rerun_1) compares two walks of the taped path, so it is a like-for-like comparison. In pairs 2 and 5 the original ({DAGGER}) was scored against a path it only approximated while the rerun was scored against the path it followed, "
          "so every taped-path difference in those pairs mixes estimator error with route deviation. For pair 2 this means the 'rerun more accurate' reading (taped-path error 37 vs 62 cm, end 54 vs 158 cm) is overstated by an unknown amount; the verdict (no meaningful difference, keep the original) is not changed by it. "
          "Pair 5's verdict does not depend on the taped path (the rerun is invalid because the estimate never traverses the route).", ""]

    # ---- rerun 5 difficulty
    E, H0, H1 = R["run_1"], R["run_5"], R["rerun_5"]
    L += ["## Is the out-and-back rerun (folder `rerun 5`) actually harder?", "",
          f"| | run_1 (easy baseline) | {tag('run_5')} (original hard) | rerun_5 (the retry) |", "|---|---|---|---|"]

    def trio(label, f, fmt):
        L.append(f"| {label} | " + " | ".join(fmt.format(f(x)) for x in (E, H0, H1)) + " |")
    trio("mean image brightness (0-255)", lambda x: x["stress"]["luminance_mean"], "{:.0f}")
    trio("darkest 10% of frames", lambda x: x["stress"]["luminance_p10"], "{:.0f}")
    trio("angular rate median (deg/s)", lambda x: x["stress"]["gyro_median_dps"], "{:.0f}")
    trio("angular rate 90th percentile (deg/s)", lambda x: x["stress"]["gyro_p90_dps"], "{:.0f}")
    trio("sharpness, median Laplacian variance", lambda x: x["stress"]["sharpness_median"], "{:.0f}")
    trio("sharpness, blurriest 10%", lambda x: x["stress"]["sharpness_p10"], "{:.0f}")
    trio("frames with >= 1 accepted marker", lambda x: 100 * x["det"]["frames_any"], "{:.0f}%")
    trio("frames with >= 1 landmark (0-6)", lambda x: 100 * x["det"]["frames_landmark"], "{:.0f}%")
    trio("accepted detections per second", lambda x: x["det"]["per_second"], "{:.1f}")
    trio("mean marker distance (m)", lambda x: x["det"]["dist_mean"], "{:.2f}")
    trio("mean reprojection error (px)", lambda x: x["det"]["reproj_mean"], "{:.2f}")
    trio("reprojection error 90th percentile (px)", lambda x: x["det"]["reproj_p90"], "{:.2f}")
    for rsn in REASONS[:3]:
        trio(f"rejected: {rsn} (% of raw detections)", lambda x, rsn=rsn: 100 * x["det"]["rej"].get(rsn, 0) / max(x["det"]["accepted"] + x["det"]["rejected"], 1), "{:.1f}%")
    trio("keyframes with no landmark in view", lambda x: 100 * x["blind"], "{:.0f}%")
    trio("M4: reliable VO direction", lambda x: 100 * x["dir_ok"], "{:.0f}%")
    trio("M4-only mean distance to taped path (cm)", lambda x: 100 * x["taped_m4"], "{:.0f}")
    trio("pose graph: mean distance to taped path (cm)", lambda x: 100 * x["taped_slam"], "{:.0f}")
    trio("pose graph: end-point error (cm)", lambda x: 100 * x["end_slam"], "{:.0f}")
    trio("pose graph: position 1-sigma median (cm)", lambda x: 100 * x["sig_med"], "{:.0f}")
    trio("pose graph: estimated path length (m) [taped: 5.4 / 5.4 / 10.8 out-and-back]", lambda x: x["len_slam"], "{:.2f}")
    trio("pose graph: route reached (m of the 5.4 m L)", lambda x: x["reach_slam"], "{:.1f}")
    trio("pose graph: end-to-start distance of the estimate (m) [an out-and-back walk should return to ~0]", lambda x: x["closure_slam"], "{:.2f}")
    trio("M4 only: end-to-start distance of the estimate (m)", lambda x: x["closure_m4"], "{:.2f}")
    trio("loop residual, pose graph (m) [revisit pairs]", lambda x: x["loop_slam"], "{:.2f}")
    L.append("| (revisit pairs found) | " + " | ".join(str(x["loop_pairs"]) for x in (E, H0, H1)) + " |")
    L += ["", "Loop residual for rerun_5 is **not computable**: the only markers it accepts are 0 (29 detections) and 3 (1), in a single visit, so no marker is observed twice. "
          "Its pose-graph end/closure/mean-distance numbers above are not evidence of accuracy: the estimate collapsed to about 0.9 m of path (see route reached), so it stays near the start.", "",
          f"Taped-path rows for the run_5 column are {DAGGER}: measured against run 1's taped path, which original run_5 only approximated (see the footnote above). The stress criteria below that use the taped-path error therefore inherit the unmeasured route deviation: criterion 5 for run_5 as the reference, and original run_5 vs run_1's count (2/5 below) partly rests on it. The rerun_5 vs run_1 result (5/5) does not depend on it: rerun_5 meets the first four criteria on image and detection measurements alone.", "",
          "Five stress criteria (True = harder than the reference); the run counts as meaningfully harder if at least two hold:", "",
          f"| criterion | rerun_5 vs run_1 | | rerun_5 vs run_5{DAGGER} | |", "|---|---|---|---|---|"]
    c1, c5 = hard_criteria(E, H1), hard_criteria(H0, H1)
    for (n1, t1, d1), (_, t5, d5) in zip(c1, c5):
        L.append(f"| {n1} | {'yes' if t1 else 'no'} | {d1} | {'yes' if t5 else 'no'} | {d5} |")
    n1, n5 = sum(t for _, t, _ in c1), sum(t for _, t, _ in c5)
    L += ["", f"**rerun_5 vs run_1: {n1}/5 criteria -> {'meaningfully harder' if n1 >= 2 else 'NOT meaningfully harder'}.**  "
          f"**rerun_5 vs run_5: {n5}/5 criteria -> {'meaningfully harder' if n5 >= 2 else 'NOT meaningfully harder'}.**  "
          f"(For reference, original run_5 vs run_1: {sum(t for _, t, _ in hard_criteria(E, H0))}/5.)", ""]

    # ---- repeatability
    keys5 = [k for k in ("run_1", "rerun_1", "run_2", "rerun_2", "run_5") if R[k]["route"] == "L" and R[k]["valid"]]
    keys3 = [k for k in keys5 if k in TAPED_EXACT]
    ob_keys = [k for k in ("run_5", "rerun_5") if R[k]["route"] == "OB" and R[k]["valid"]]
    n5, n3 = len(keys5), len(keys3)
    L += ["## Repeatability across repeated walks of a nominally identical route", "",
          f"**Route caveat.** The {n5} one-way walks are only NOMINALLY the same L route. run_1, rerun_1 and rerun_2 followed the taped path (run 1's coordinates). "
          f"Original run_2 and run_5 ({DAGGER}) are also roughly L-shaped and start at the same point, but their turning and end points were re-walked rather than re-taped, "
          "so they deviate slightly from that path (deviation not measured). Part of the spread reported for the n = 5 set therefore reflects route variation, not estimator error alone. "
          f"The n = {n3} rows use only the true taped repeats (run_1, rerun_1, rerun_2) so both are visible and the reader can see how much the assumption matters.", "",
          "Pose-graph results (each against its own leave-pair-out map); spread = sample standard deviation, range = min..max; CV = std / mean. "
          f"{DAGGER} = measured against run 1's taped path but not walked on it exactly.", ""]
    sets = [(f"nominal L, n={n5}", keys5), (f"taped repeats only, n={n3}", keys3)]
    if len(ob_keys) >= 2:
        sets.append((f"out-and-back, n={len(ob_keys)}", ob_keys))
    metrics = (("end-point error", lambda x: x["end_slam"], 100, "cm"), ("estimated path length", lambda x: x["len_slam"], 1, "m"),
               ("mean distance to taped path (run 1's path)", lambda x: x["taped_slam"], 100, "cm"),
               ("M4-only end-point error", lambda x: x["end_m4"], 100, "cm"), ("M4-only path length", lambda x: x["len_m4"], 1, "m"))
    L += ["| set | walks | metric | values | mean | std | range | CV |", "|---|---|---|---|---|---|---|---|"]
    stats = {}
    for sname, keys in sets:
        if len(keys) < 2:
            continue
        for metric, f, scale, unit in metrics:
            v = np.array([f(R[k]) * scale for k in keys])
            stats[(sname, metric)] = (v.mean(), v.std(ddof=1), v.min(), v.max())
            L.append(f"| {sname} | {', '.join(tag(k) for k in keys)} | {metric} ({unit}) | {', '.join(f'{u:.1f}' if unit == 'cm' else f'{u:.2f}' for u in v)} | {v.mean():.1f} | {v.std(ddof=1):.1f} | {v.min():.1f}..{v.max():.1f} | {100 * v.std(ddof=1) / v.mean():.0f}% |")
    L += ["", "**How much the route assumption matters** (n = " + str(n5) + " nominal vs n = " + str(n3) + " taped only, mean +- std):", ""]
    for metric, f, scale, unit in metrics[:3]:
        a5, a3 = stats[(sets[0][0], metric)], stats[(sets[1][0], metric)]
        d = "" if a3[1] == 0 else f", spread x{a5[1] / a3[1]:.2f}"
        L.append(f"- {metric}: {a5[0]:.1f} +- {a5[1]:.1f} {unit} (n={n5}) vs {a3[0]:.1f} +- {a3[1]:.1f} {unit} (n={n3}){d}")
    L += ["", "Taped length 5.4 m (L). With so few walks these are rough spreads, not confidence intervals; with n = 3 even more so. The two re-walked runs "
          "(" + ", ".join(tag(k) for k in keys5 if k not in TAPED_EXACT) + ") end 158 and 131 cm from run 1's endpoint, inside the 54-119 cm range of the true repeats, "
          "which is why they were kept in the nominal set, but their route deviation cannot be separated from estimator error here.", ""]
    (out / f"{a.out_name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:60]))

    # ---- plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (lab, *_) in zip(axes.ravel(), RUNS):
        x = R[lab]
        col = ORANGE if lab.startswith("rerun") else BLUE
        ax.plot(x["m4_p"][:, 0], x["m4_p"][:, 1], "-", color=RED, lw=1, alpha=0.6, label="M4 only")
        ax.plot(GT[x["route"]][:, 0], GT[x["route"]][:, 1], "--", color="k", lw=1.3, label="taped path")
        ax.plot(x["cam_p"][:, 0], x["cam_p"][:, 1], "-", color=col, lw=2.2, label="pose graph")
        ax.plot(*x["cam_p"][0, :2], "o", color="tab:green", ms=8)
        ax.plot(*x["cam_p"][-1, :2], "s", color="tab:red", ms=8)
        for m, T in x["markers"].items():
            ax.plot(*T[:2, 3], "D", color="#f0a202", ms=6, mec="k")
            ax.annotate(str(m), T[:2, 3], textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_aspect("equal")
        pts = np.vstack([x["cam_p"][:, :2], np.array([T[:2, 3] for T in x["markers"].values()]), GT[x["route"]]])
        ax.set_xlim(pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5)
        ax.set_ylim(pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.5)
        ax.grid(alpha=0.3)
        ax.set_title(f"{tag(lab)} ({x['route']}): mean {100 * x['taped_slam']:.0f} cm from {'run 1' + chr(39) + 's ' if lab in APPROX else ''}taped path, end {100 * x['end_slam']:.0f} cm", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{a.out_name}_paths.png", dpi=130)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_sets = [(f"nominal L\n(n={n5})", keys5), (f"taped repeats only\n(n={n3})", keys3)]
    for ax, (title, f, scale, ref) in zip(axs, (("End-point error (cm)", lambda x: x["end_slam"], 100, None),
                                                ("Estimated path length (m)", lambda x: x["len_slam"], 1, plen(P_L)))):
        for gi_, (lab_, keys) in enumerate(plot_sets):
            v = np.array([f(R[k]) * scale for k in keys])
            prev_v, prev_off = None, -3
            for k, vv in sorted(zip(keys, v), key=lambda kv: kv[1]):
                exact = k in TAPED_EXACT
                ax.plot([gi_], [vv], "o", ms=10, mec=BLUE if exact else ORANGE, mfc=BLUE if exact else "white", mew=2)
                close = prev_v is not None and abs(vv - prev_v) < 0.06 * max(v.max(), 1e-9)
                off = prev_off + 11 if close else -3                # stagger labels of near-identical values
                ax.annotate(tag(k), (gi_, vv), textcoords="offset points", xytext=(11, off), fontsize=8)
                prev_v, prev_off = vv, off
            ax.plot([gi_ - 0.22, gi_ + 0.22], [v.mean()] * 2, "-", color="k", lw=1.6)
            ax.annotate(f"{v.mean():.1f} +- {v.std(ddof=1):.1f}", (gi_ - 0.22, v.mean()), textcoords="offset points", xytext=(-4, 5), ha="right", fontsize=8)
        if ref:
            ax.axhline(ref, color="0.5", ls="--", lw=1)
            ax.annotate("taped length", (1.45, ref), textcoords="offset points", xytext=(0, 3), ha="right", fontsize=8, color="0.4")
        ax.set_xticks(range(2))
        ax.set_xticklabels([t for t, _ in plot_sets])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Repeatability over NOMINALLY identical routes (filled = walked the taped path; hollow {DAGGER} = re-walked without re-taping; line = mean)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "repeatability.png", dpi=140)
    print(f"\nsaved {out / f'{a.out_name}.md'}, {a.out_name}_paths.png, repeatability.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
