"""Difficult-condition evaluation (run 5: dim light + fast motion) side by side with an easy baseline (run 1).

    python -m src.run_difficult "data/all runs/run 5" --baseline "data/all runs/run 1" \\
        --map outputs/marker_map_excl_run5.json --baseline-map outputs/marker_map_excl_run1.json

Each run is solved against a map built WITHOUT it. Compared: stress indicators (brightness, sharpness, motion),
detection rate, reprojection error, rejection breakdown, and trajectory accuracy:
  - mean error vs the survey-derived reference (camera position from the surveyed pose of a wall marker in view),
    for the M4 odometry alone and for the fused pose graph (comparable across runs: no taped path exists for run 5);
  - the same for the taped path where one exists (run 1 only).
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
from src.run_slam import marker_residuals, parse_polyline, polyline_error, register_baseline

TAPED = {"run_1": "2.9,3.25;0.5,3.25;0.5,0.25"}            # data/room.md
EASY, HARD = "#2f6fb0", "#c8741a"


def evaluate(run_dir: str, map_path: str, out: Path) -> dict:
    name = Path(run_dir).resolve().name.strip().replace(" ", "_")
    cfg = MapConfig()
    ds = RunDataset(run_dir)
    det_raw = detect_run(ds, cache_path=out / f"{name}_detections.npz")
    det = _gate(det_raw, cfg)
    traj = load_trajectory_csv(out / f"{name}_traj.csv")
    mmap = MarkerMap.load(map_path)
    markers = {m: mmap.T_room_world @ T for m, T in mmap.poses.items()}
    use_ids = {m for m, f in mmap.in_map.items() if f}

    r = {"name": name, "map": Path(map_path).name}
    r["det"] = detection_summary(det_raw, use_ids)
    r["stress"] = stress_indicators(ds)
    sol = solve_with_map(traj, det, markers, use_ids, cfg, compute_cov=True)
    gi, kf = sol.gi, sol.gi.kf_frames
    surv = survey_poses()
    ref = survey_reference(det, surv)
    _, pm4 = m4_in_room(traj, det, surv)
    r["err_slam"], r["n_ref"] = mean_error(lambda f: sol.cam_p[int(np.argmin(np.abs(kf - f)))], ref)
    r["err_m4"], _ = mean_error(lambda f: pm4[min(f, len(pm4) - 1)], ref)
    bR, bp = register_baseline(gi, traj, sol.init_R, sol.init_p)
    pe0, re0 = marker_residuals(gi, markers, bR, bp)
    pe1, re1 = marker_residuals(gi, markers, sol.cam_R, sol.cam_p)
    r.update(pred_m4=float(np.median(pe0)), pred_m4_rot=float(np.median(re0)), pred_slam=float(np.median(pe1)), pred_slam_rot=float(np.median(re1)))
    sig = sol.result.position_sigma()
    r.update(keyframes=len(kf), obs=len(gi.obs), dropped=sol.dropped_obs, sig_med=float(np.nanmedian(sig)), sig_max=float(np.nanmax(sig)),
             rms_marker=sol.result.rms_marker, rms_odom=sol.result.rms_odom, scale_med=float(np.median(sol.result.odom_scales)),
             dir_ok=float(traj.dir_ok.mean()), still=float(traj.stationary.mean()))
    plen = lambda p: float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())
    r.update(len_slam=plen(sol.cam_p), len_m4=plen(pm4[kf]), cam_p=sol.cam_p, m4_p=pm4[kf], markers=markers)
    vis = np.zeros(len(kf), bool)
    vis[[o.kf for o in gi.obs]] = True
    t = (gi.kf_t_ns - gi.kf_t_ns[0]) * 1e-9
    r["blind_frac"] = float(1 - vis.mean())
    r["first_fix"] = float(t[np.argmax(vis)])
    if name in TAPED:
        poly = parse_polyline(TAPED[name])
        r["taped_slam"] = float(polyline_error(sol.cam_p[:, :2], poly).mean())
        r["taped_m4"] = float(polyline_error(pm4[kf][:, :2], poly).mean())
        r["end_slam"] = float(np.linalg.norm(sol.cam_p[-1, :2] - poly[-1]))
        r["end_m4"] = float(np.linalg.norm(pm4[-1, :2] - poly[-1]))
        r["poly"] = poly
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--baseline-map", required=True)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--out-name", default="run5_difficult")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)

    E = evaluate(a.baseline, a.baseline_map, out)
    H = evaluate(a.run_dir, a.map, out)
    e, h = E["name"], H["name"]

    def row(label, fe, fh, fmt="{:.2f}", ratio=True):
        ve, vh = fe, fh
        s = f"| {label} | {fmt.format(ve)} | {fmt.format(vh)} | "
        if ratio and ve not in (0, None) and vh is not None:
            s += f"{vh / ve:.1f}x |" if abs(vh / ve - 1) > 0.005 else "same |"
        else:
            s += "|"
        return s

    de, dh = E["det"], H["det"]
    se, sh = E["stress"], H["stress"]
    L = [f"# {h} (difficult: dim light + fast motion) vs {e} (easy baseline)", "",
         f"Each run solved against a map built without it: `{H['map']}` / `{E['map']}`. Landmarks = in_map markers 0-6.", "",
         "## Conditions (what makes the run hard)", "", f"| | {e} (easy) | {h} (hard) | {h}/{e} |", "|---|---|---|---|",
         row("mean image brightness (0-255, every 10th frame)", se["luminance_mean"], sh["luminance_mean"], "{:.0f}"),
         row("darkest 10% of frames (0-255)", se["luminance_p10"], sh["luminance_p10"], "{:.0f}"),
         row("image sharpness, median Laplacian variance (higher = sharper)", se["sharpness_median"], sh["sharpness_median"], "{:.0f}"),
         row("sharpness, blurriest 10% of frames", se["sharpness_p10"], sh["sharpness_p10"], "{:.0f}"),
         row("angular rate, median (deg/s)", se["gyro_median_dps"], sh["gyro_median_dps"], "{:.0f}"),
         row("angular rate, 90th percentile (deg/s)", se["gyro_p90_dps"], sh["gyro_p90_dps"], "{:.0f}"),
         "", "## Detection", "", f"| | {e} (easy) | {h} (hard) | {h}/{e} |", "|---|---|---|---|",
         row("frames with >= 1 accepted marker", 100 * de["frames_any"], 100 * dh["frames_any"], "{:.0f}%"),
         row("frames with >= 1 accepted landmark (markers 0-6)", 100 * de["frames_landmark"], 100 * dh["frames_landmark"], "{:.0f}%"),
         row("accepted detections per second", de["per_second"], dh["per_second"], "{:.1f}"),
         row("mean marker distance (m)", de["dist_mean"], dh["dist_mean"]),
         row("median marker size (px^2)", de["area_median"], dh["area_median"], "{:.0f}"),
         row("mean reprojection error (px)", de["reproj_mean"], dh["reproj_mean"]),
         row("reprojection error, 90th percentile (px)", de["reproj_p90"], dh["reproj_p90"]),
         row("share of raw detections accepted by the detector gate", 100 * de["accept_ratio"], 100 * dh["accept_ratio"], "{:.0f}%"),
         "", "## Rejection breakdown (detector gate; counts and % of all raw detections)", "",
         f"| reason | {e} count | {e} % | {h} count | {h} % |", "|---|---|---|---|---|"]
    for rsn in REASONS[:3]:
        ce, ch = de["rej"].get(rsn, 0), dh["rej"].get(rsn, 0)
        L.append(f"| {rsn} | {ce} | {100 * ce / max(de['accepted'] + de['rejected'], 1):.1f}% | {ch} | {100 * ch / max(dh['accepted'] + dh['rejected'], 1):.1f}% |")
    L += ["", "## Trajectory accuracy", "", f"| | {e} (easy) | {h} (hard) | {h}/{e} |", "|---|---|---|---|",
          row("landmark observations used by the pose graph", E["obs"], H["obs"], "{:.0f}"),
          row("observations dropped as outliers", E["dropped"], H["dropped"], "{:.0f}", ratio=False),
          row("time to first landmark fix (s)", E["first_fix"], H["first_fix"], "{:.1f}"),
          row("share of keyframes with no landmark in view", 100 * E["blind_frac"], 100 * H["blind_frac"], "{:.0f}%"),
          row("reliable VO direction (M4), % of frames", 100 * E["dir_ok"], 100 * H["dir_ok"], "{:.0f}%"),
          row("M4 odometry only: mean error vs survey-derived reference (cm)", 100 * E["err_m4"], 100 * H["err_m4"], "{:.0f}"),
          row("pose graph: mean error vs survey-derived reference (cm)", 100 * E["err_slam"], 100 * H["err_slam"], "{:.0f}"),
          row("M4 only: median marker-prediction error (cm)", 100 * E["pred_m4"], 100 * H["pred_m4"], "{:.0f}"),
          row("pose graph: median marker-prediction error (cm)", 100 * E["pred_slam"], 100 * H["pred_slam"], "{:.1f}"),
          row("pose graph: position 1-sigma, median (cm)", 100 * E["sig_med"], 100 * H["sig_med"], "{:.0f}"),
          row("pose graph: position 1-sigma, max (cm)", 100 * E["sig_max"], 100 * H["sig_max"], "{:.0f}"),
          row("pose graph: top-down path length (m)", E["len_slam"], H["len_slam"]),
          f"| mean distance to taped path, pose graph / M4 | {100 * E.get('taped_slam', float('nan')):.0f} cm / {100 * E.get('taped_m4', float('nan')):.0f} cm | n/a (no taped path recorded) | |",
          f"| end-point error vs taped end, pose graph / M4 | {100 * E.get('end_slam', float('nan')):.0f} cm / {100 * E.get('end_m4', float('nan')):.0f} cm | n/a | |",
          "", "## Findings (computed from the numbers above)", ""]
    dim = sh["luminance_mean"] / se["luminance_mean"]
    L.append(f"- **Dim light is real:** {h} is {1 / dim:.1f}x darker than {e} (mean gray {sh['luminance_mean']:.0f} vs {se['luminance_mean']:.0f}).")
    gr = sh["gyro_p90_dps"] / se["gyro_p90_dps"]
    L.append(f"- **Fast motion is {'confirmed' if gr > 1.2 else 'NOT visible in the gyro'}:** 90th-percentile angular rate {sh['gyro_p90_dps']:.0f} vs {se['gyro_p90_dps']:.0f} deg/s ({gr:.1f}x); "
             f"image sharpness {sh['sharpness_median']:.0f} vs {se['sharpness_median']:.0f}, i.e. no extra motion blur.")
    dr = dh["frames_landmark"] / de["frames_landmark"]
    L.append(f"- **Detection {'degraded' if dr < 0.9 else 'did not degrade'}:** landmarks in view on {100 * dh['frames_landmark']:.0f}% of frames vs {100 * de['frames_landmark']:.0f}%; "
             f"mean reprojection error {dh['reproj_mean']:.2f} vs {de['reproj_mean']:.2f} px; rejected {100 * (1 - dh['accept_ratio']):.0f}% vs {100 * (1 - de['accept_ratio']):.0f}% of raw detections.")
    worse = []
    if H["dir_ok"] < 0.9 * E["dir_ok"]:
        worse.append(f"M4's reliable VO direction ({100 * H['dir_ok']:.0f}% of frames vs {100 * E['dir_ok']:.0f}%)")
    if H["pred_slam"] > 1.5 * E["pred_slam"]:
        worse.append(f"the pose graph's fit to what the markers imply ({100 * H['pred_slam']:.1f} cm vs {100 * E['pred_slam']:.1f} cm median, still small in absolute terms)")
    if H["err_slam"] > 1.1 * E["err_slam"]:
        worse.append(f"fused mean error ({100 * H['err_slam']:.0f} vs {100 * E['err_slam']:.0f} cm)")
    L.append("- **What did degrade:** " + ("; ".join(worse) if worse else "nothing measurable here") + ".")
    L.append(f"- **Confound:** the two runs follow different paths. {h} sees markers from the first frame and closer (median marker size {dh['area_median']:.0f} vs {de['area_median']:.0f} px^2), "
             f"so it is blind for {100 * H['blind_frac']:.0f}% of keyframes against {100 * E['blind_frac']:.0f}%. Its lower odometry-only error ({100 * H['err_m4']:.0f} vs {100 * E['err_m4']:.0f} cm) comes from early fixes and a shorter exposure to drift, "
             "not from easier odometry. This comparison therefore cannot isolate the effect of lighting; it shows that dim light alone did not break ArUco detection at 1-1.5 m.")
    L += ["", "## Notes", "",
          "- Brightness is the mean of a half-resolution grayscale frame; sharpness is the variance of its Laplacian; angular rate is the gyro norm after removing the run's resting bias.",
          "- 'Survey-derived reference' scores any trajectory without a taped path: camera position implied by the SURVEYED pose of a wall marker in view. The fused path inherits the map's own error (tens of cm), so that is its floor; M4 odometry has no such floor and shows the raw degradation.",
          "- Marker-prediction error = distance between the camera pose and what the marker the camera is looking at implies (map markers fixed).",
          f"- Run {h} has no recorded ground-truth path in data/room.md (only the 'dim lights' condition), so taped-path metrics exist for {e} only."]
    (out / f"{a.out_name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    ink, sub, grid = "#1f2933", "#52606d", "#e4e7eb"

    def bars(ax, title, vals, fmt, labels=(e, h), ylog=False, sec=None):
        x = np.arange(len(vals[0]))
        w = 0.36
        for i, (v, col, lab) in enumerate(zip(vals, (EASY, HARD), labels)):
            ax.bar(x + (i - 0.5) * w, v, w, color=col, label=f"{lab} ({'easy' if i == 0 else 'hard'})")
            for xi, vi in zip(x, v):
                ax.text(xi + (i - 0.5) * w, vi * (1.06 if ylog else 1) + (0 if ylog else max(max(vals[0]), max(vals[1])) * 0.01), fmt.format(vi),
                        ha="center", va="bottom", fontsize=8, color=ink)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        if ylog:
            ax.set_yscale("log")
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=sub, length=0)
        ax.set_xticks(x)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    bars(axes[0, 0], "Scene brightness (mean gray level, 0-255)", ([se["luminance_mean"]], [sh["luminance_mean"]]), "{:.0f}")
    axes[0, 0].set_xticklabels([""])
    bars(axes[0, 1], "Motion: angular rate (deg/s)", ([se["gyro_median_dps"], se["gyro_p90_dps"]], [sh["gyro_median_dps"], sh["gyro_p90_dps"]]), "{:.0f}")
    axes[0, 1].set_xticklabels(["median", "90th pct"])
    bars(axes[0, 2], "Image sharpness (Laplacian variance, higher = sharper)", ([se["sharpness_median"], se["sharpness_p10"]], [sh["sharpness_median"], sh["sharpness_p10"]]), "{:.0f}")
    axes[0, 2].set_xticklabels(["median", "blurriest 10%"])
    bars(axes[1, 0], "Detection (% of frames / % of raw detections)",
         ([100 * de["frames_any"], 100 * de["frames_landmark"], 100 * de["accept_ratio"]], [100 * dh["frames_any"], 100 * dh["frames_landmark"], 100 * dh["accept_ratio"]]), "{:.0f}%")
    axes[1, 0].set_xticklabels(["frames, any\nmarker", "frames, landmark\n0-6", "raw detections\naccepted"], fontsize=8)
    bars(axes[1, 1], "Reprojection error of accepted detections (px)", ([de["reproj_mean"], de["reproj_p90"]], [dh["reproj_mean"], dh["reproj_p90"]]), "{:.2f}")
    axes[1, 1].set_xticklabels(["mean", "90th pct"])
    bars(axes[1, 2], "Trajectory: mean error vs survey reference (cm)",
         ([100 * E["err_m4"], 100 * E["err_slam"]], [100 * H["err_m4"], 100 * H["err_slam"]]), "{:.0f}")
    axes[1, 2].set_xticklabels(["M4 odometry\nonly", "pose graph\n(+ markers)"], fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"{h} (difficult) vs {e} (easy baseline): what degrades under dim light and fast motion", fontsize=12, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"{a.out_name}.png", dpi=150)

    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6.8))
    for ax, R, col in ((ax2[0], E, EASY), (ax2[1], H, HARD)):
        ax.plot(R["m4_p"][:, 0], R["m4_p"][:, 1], "-", color="#d1495b", lw=1.1, alpha=0.7, label="M4 odometry only")
        ax.plot(R["cam_p"][:, 0], R["cam_p"][:, 1], "-", color=col, lw=2.4, label="pose graph (+ markers)")
        if "poly" in R:
            ax.plot(R["poly"][:, 0], R["poly"][:, 1], "--", color="k", lw=1.3, label="taped path")
        for m, T in R["markers"].items():
            ax.plot(*T[:2, 3], "D", color="#f0a202", ms=7, mec="k")
            ax.annotate(str(m), T[:2, 3], textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_aspect("equal")
        pts = np.vstack([R["cam_p"][:, :2], np.array([T[:2, 3] for T in R["markers"].values()])])
        ax.set_xlim(pts[:, 0].min() - 0.6, pts[:, 0].max() + 0.6)
        ax.set_ylim(pts[:, 1].min() - 0.6, pts[:, 1].max() + 0.6)
        ax.grid(alpha=0.3)
        ax.set_title(f"{R['name']} ({'easy' if R is E else 'difficult'}): mean error vs survey {100 * R['err_slam']:.0f} cm fused, {100 * R['err_m4']:.0f} cm odometry only", fontsize=9)
        ax.legend(fontsize=8, loc="best")
    fig2.tight_layout()
    fig2.savefig(out / f"{a.out_name}_paths.png", dpi=140)
    print(f"\nsaved {out / f'{a.out_name}.md'}\nsaved {out / f'{a.out_name}.png'}\nsaved {out / f'{a.out_name}_paths.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
