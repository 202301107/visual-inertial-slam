"""Streamlit front end for the visual-inertial SLAM pipeline. Wiring only: no new estimation.

    streamlit run src/interface/app.py

Tabs: run & sensor status / detected markers / trajectory & map / diagnostics (saved M6 evidence).
Results of "Run detection + SLAM" are saved to outputs/<run>_app_slam.npz (the CLI's *_slam.* files are left alone).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_markers import DEFAULT_CALIB_PATH, load_calibration
from src.detection.detect_run import RunDetections, detect_run
from src.estimation.slam_core import solve_with_map
from src.mapping.marker_map import MapConfig, MarkerMap, _gate, load_ground_truth
from src.odometry.trajectory import TrajectoryConfig, build_trajectory, load_trajectory_csv
from src.run_slam import parse_polyline, register_baseline
from src.run_trajectory import run_vo

DATA = ROOT / "data" / "all runs"
OUT = ROOT / "outputs"
GT_CSV = ROOT / "data" / "markers.csv"
RUN1_TAPED_PATH = "2.9,3.25;0.5,3.25;0.5,0.25"      # data/room.md, run 1 ground-truth path
REASONS = ["unknown_id", "too_small", "ambiguous", "behind_camera"]

st.set_page_config(page_title="Visual-inertial SLAM", layout="wide")


# ----------------------------------------------------------------------------- helpers
def run_key(path: Path) -> str:
    return path.resolve().name.strip().replace(" ", "_")


@st.cache_data(show_spinner=False)
def list_runs() -> list[str]:
    if not DATA.is_dir():
        return []
    return [d.name for d in sorted(DATA.iterdir()) if d.is_dir() and any(d.rglob("VID_*.mp4"))]


@st.cache_data(show_spinner=False)
def sensor_status(run_dir: str) -> dict:
    ds = RunDataset(run_dir)
    t0 = int(ds.frame_t_ns[0])
    rate = lambda t: 1e9 / float(np.median(np.diff(t)))
    return {
        "video": ds.video_path.name, "size": ds.image_size, "frames_ts": len(ds.frame_t_ns), "frames_video": ds.n_video_frames,
        "fps": rate(ds.frame_t_ns), "duration": (int(ds.frame_t_ns[-1]) - t0) * 1e-9,
        "accel_n": len(ds.accel), "accel_hz": rate(ds.accel.t_ns), "gyro_n": len(ds.gyro), "gyro_hz": rate(ds.gyro.t_ns),
        "accel_off": ((int(ds.accel.t_ns[0]) - t0) * 1e-9, (int(ds.accel.t_ns[-1]) - t0) * 1e-9),
        "gyro_off": ((int(ds.gyro.t_ns[0]) - t0) * 1e-9, (int(ds.gyro.t_ns[-1]) - t0) * 1e-9),
        "video_path": str(ds.video_path), "frame_t": ds.frame_t_ns.tolist(),
    }


def load_det(name: str) -> RunDetections | None:
    p = OUT / f"{name}_detections.npz"
    return RunDetections.load(p) if p.is_file() else None


def map_choices() -> list[str]:
    return sorted(p.name for p in OUT.glob("marker_map*.json"))


def default_map_for(name: str, choices: list[str]) -> str | None:
    """Prefer a map built WITHOUT this run (honest), else the pooled map."""
    leave_out = f"marker_map_excl_run{name.split('_')[-1]}.json"
    if leave_out in choices:
        return leave_out
    return "marker_map.json" if "marker_map.json" in choices else (choices[0] if choices else None)


def save_app_result(name, sol, base_p, map_name):
    cov = sol.result.cov if sol.result.cov is not None else np.empty(0)
    np.savez(OUT / f"{name}_app_slam.npz", t_ns=sol.gi.kf_t_ns, p=sol.cam_p, R_wc=sol.cam_R, cov=cov, baseline_p=base_p,
             frames=sol.gi.kf_frames, map_name=map_name, odom_scales=sol.result.odom_scales,
             n_obs=len(sol.gi.obs), dropped=sol.dropped_obs, n_loops=len(sol.loops))


@st.cache_data(show_spinner=False)
def load_app_result(path: str, mtime: float):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def traj_figure(res, markers_room, gt, gt_path, show_m4, show_ellipses, show_gt_markers):
    p, base = res["p"], res["baseline_p"]
    t = (res["t_ns"] - res["t_ns"][0]) * 1e-9
    fig, ax = plt.subplots(figsize=(8.5, 8))
    if show_m4:
        ax.plot(base[:, 0], base[:, 1], "-", color="#d1495b", lw=1.4, alpha=0.85, label="M4 odometry only (no markers)")
    if gt_path is not None:
        ax.plot(gt_path[:, 0], gt_path[:, 1], "--", color="k", lw=1.5, label="taped ground-truth path")
    seg = np.stack([p[:-1, :2], p[1:, :2]], axis=1)
    lc = LineCollection(seg, cmap="viridis", linewidth=2.6)
    lc.set_array(0.5 * (t[:-1] + t[1:]))
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, shrink=0.7, label="time (s)")
    cov = res["cov"]
    if show_ellipses and cov.size:
        for k in range(0, len(p), max(len(p) // 25, 1)):
            w, V = np.linalg.eigh(cov[k, 3:5, 3:5])
            ax.add_patch(Ellipse(p[k, :2], 4 * np.sqrt(max(w[1], 0)), 4 * np.sqrt(max(w[0], 0)),
                                 angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])), fill=False, ec="#2f6fb0", lw=0.9, alpha=0.7))
        ax.plot([], [], color="#2f6fb0", lw=0.9, label="2-sigma position ellipse")
    if show_gt_markers:
        for m, g in gt.items():
            ax.plot(*g["pos"][:2], "x", color="0.4", ms=6)
        ax.plot([], [], "x", color="0.4", label="surveyed marker (markers.csv)")
    for m, T in sorted(markers_room.items()):
        ax.plot(*T[:2, 3], "D", color="#f0a202", ms=8, mec="k")
        ax.annotate(str(m), T[:2, 3], textcoords="offset points", xytext=(6, 6), fontsize=10, weight="bold")
    ax.plot([], [], "D", color="#f0a202", mec="k", label="mapped marker")
    ax.plot(*p[0, :2], "o", color="tab:green", ms=10, label="start")
    ax.plot(*p[-1, :2], "s", color="tab:red", ms=9, label="end")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.08)
    ax.grid(alpha=0.3)
    ax.set_xlabel("X (m), room frame")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def path_len(p):
    return float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())


def show_md_section(path: Path, heading_start: str, make_cmd: str):
    """Render only the '## <heading_start>...' section of a markdown file (up to the next '## ')."""
    if not path.is_file():
        st.info(f"`{path.name}` not found. Create it with `{make_cmd}`.")
        return
    lines, keep, out = path.read_text(encoding="utf-8").splitlines(), False, []
    for ln in lines:
        if ln.startswith("## "):
            keep = ln[3:].startswith(heading_start)
        if keep:
            out.append(ln)
    if out:
        st.markdown("\n".join(out))
    else:
        st.info(f"No section starting with '{heading_start}' in `{path.name}`.")


def show_file(path: Path, kind: str, make_cmd: str):
    if not path.is_file():
        st.info(f"`{path.name}` not found. Create it with `{make_cmd}`.")
    elif kind == "md":
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.image(str(path), width="stretch")


# ----------------------------------------------------------------------------- sidebar
st.title("Visual-inertial SLAM with fiducial landmark anchors")
runs = list_runs()
if not runs:
    st.error(f"No run folders found under `{DATA}`.")
    st.stop()

with st.sidebar:
    st.header("Dataset")
    run_name = st.selectbox("Run folder (data/all runs)", runs)
    run_dir = DATA / run_name
    name = run_key(run_dir)
    choices = map_choices()
    st.header("Marker map")
    if choices:
        default = default_map_for(name, choices)
        map_name = st.selectbox("Saved map", choices, index=choices.index(default),
                                help="A `marker_map_excl_runN.json` map was built without run N, so evaluating run N against it is honest. "
                                     "`marker_map.json` pools all runs (it includes the selected run).")
        st.caption("Built without this run." if map_name == f"marker_map_excl_run{name.split('_')[-1]}.json" else
                   "Includes this run's own detections (optimistic)." if map_name == "marker_map.json" else "")
    else:
        map_name = None
        st.warning("No outputs/marker_map*.json found. Build one: `python -m src.mapping.marker_map \"data/all runs\"`")
    st.header("Options")
    use_floor = st.checkbox("Also use floor markers 7-9 as landmarks", value=False,
                            help="Off by default: they are scoring points, not map landmarks.")
    loop_cl = st.checkbox("Add loop-closure edges (re-observed markers)", value=False)
    gt_text = st.text_input("Taped ground-truth path (x,y;x,y;...)", value=RUN1_TAPED_PATH if name == "run_1" else "",
                            help="Optional. Room metres, from data/room.md (run 1 prefilled). Only run 1, rerun 1 and rerun 2 walked this taped L exactly; "
                                 "original runs 2 and 5 only approximated it (turning/end points re-walked, not re-taped).")

# ----------------------------------------------------------------------------- tabs
tab_run, tab_markers, tab_traj, tab_diag = st.tabs(["1. Run & sensors", "2. Detected markers", "3. Trajectory & map", "4. Diagnostics (M6 evidence)"])

# ---- tab 1 --------------------------------------------------------------------------------------------------
with tab_run:
    stt = sensor_status(str(run_dir))
    st.subheader(f"Camera / sensor status: {run_name}")
    c = st.columns(4)
    c[0].metric("Video resolution", f"{stt['size'][0]} x {stt['size'][1]}")
    c[1].metric("Frames (timestamps)", stt["frames_ts"])
    c[2].metric("Frame rate", f"{stt['fps']:.1f} fps")
    c[3].metric("Duration", f"{stt['duration']:.1f} s")
    c = st.columns(4)
    c[0].metric("Accelerometer", f"{stt['accel_n']:,} samples", f"{stt['accel_hz']:.0f} Hz", delta_color="off")
    c[1].metric("Gyroscope", f"{stt['gyro_n']:,} samples", f"{stt['gyro_hz']:.0f} Hz", delta_color="off")
    c[2].metric("IMU starts (vs first frame)", f"{stt['accel_off'][0] * 1000:+.0f} ms")
    c[3].metric("IMU ends (vs last frame)", f"{(stt['accel_off'][1] - stt['duration']) * 1000:+.0f} ms")
    st.caption(f"Video `{stt['video']}`; accel/gyro share one clock with the frame timestamps. "
               f"Calibration: `{Path(DEFAULT_CALIB_PATH).name}`.")
    if stt["frames_ts"] != stt["frames_video"]:
        st.warning(f"The video reports {stt['frames_video']} frames but there are {stt['frames_ts']} timestamps; extra timestamps are dropped.")

    st.subheader("Processing")
    det_cache = OUT / f"{name}_detections.npz"
    traj_csv = OUT / f"{name}_traj.csv"
    st.write(f"- detections cache: {'**found**' if det_cache.is_file() else 'not found (will be computed, ~1-2 min)'}  \n"
             f"- M4 trajectory (visual-inertial, no markers): {'**found**' if traj_csv.is_file() else 'not found (will be computed, a few minutes)'}")
    go = st.button("Run detection + SLAM", type="primary", disabled=map_name is None)
    if go:
        try:
            ds = RunDataset(run_dir)
            K, dist, _ = load_calibration()
            bar, msg = st.progress(0.0), st.empty()

            msg.write("Step 1/3: marker detection ...")
            det = detect_run(ds, cache_path=det_cache, progress=lambda i, n: bar.progress(min(i / n, 1.0) * 0.4)) \
                if not det_cache.is_file() else RunDetections.load(det_cache)
            bar.progress(0.4)

            if traj_csv.is_file():
                traj = load_trajectory_csv(traj_csv)
            else:
                msg.write("Step 2/3: visual odometry + IMU (M4 trajectory) ...")
                steps = run_vo(ds, K, dist, 1.0, progress=lambda i, n: bar.progress(0.4 + min(i / n, 1.0) * 0.45))
                traj = build_trajectory(ds, steps, K, TrajectoryConfig())
                traj.save_csv(traj_csv)
            bar.progress(0.85)

            msg.write("Step 3/3: pose graph (odometry + marker factors) ...")
            mmap = MarkerMap.load(OUT / map_name)
            markers = {m: mmap.T_room_world @ T for m, T in mmap.poses.items()}
            use_ids = set(mmap.poses) if use_floor else ({m for m, f in mmap.in_map.items() if f} or set(mmap.poses))
            sol = solve_with_map(traj, _gate(det, MapConfig()), markers, use_ids, MapConfig(), loop_closure=loop_cl,
                                 loop_min_gap_s=3.0, compute_cov=True)
            if sol is None:
                raise RuntimeError("No landmark from the selected map is detected in this run.")
            base_R, base_p = register_baseline(sol.gi, traj, sol.init_R, sol.init_p)
            save_app_result(name, sol, base_p, map_name)
            bar.progress(1.0)
            msg.success(f"Done: {len(sol.gi.kf_frames)} keyframes, {len(sol.gi.obs)} marker observations "
                        f"({sol.dropped_obs} outliers dropped), {len(sol.loops)} loop-closure edges. See the other tabs.")
            st.cache_data.clear()
        except Exception as e:                      # keep the demo alive and say what happened
            st.error(f"Processing failed: {e}")

# ---- tab 2 --------------------------------------------------------------------------------------------------
with tab_markers:
    det = load_det(name)
    st.subheader("Detected markers")
    if det is None:
        st.info("No detections for this run yet: press **Run detection + SLAM** in tab 1.")
    else:
        rows = []
        rej_known = det.rej_marker is not None
        ids = sorted(set(det.marker_id.tolist()) | (set(det.rej_marker.tolist()) if rej_known else set()))
        for m in ids:
            s = det.marker_id == m
            rej = "n/a" if not rej_known else (", ".join(f"{r}={int(((det.rej_marker == m) & (det.rej_reason == r)).sum())}" for r in REASONS
                                                           if ((det.rej_marker == m) & (det.rej_reason == r)).any()) or "-")
            rows.append({"ID": m, "detections": int(s.sum()),
                         "mean distance (m)": round(float(det.distance[s].mean()), 2) if s.any() else None,
                         "mean reprojection error (px)": round(float(det.reproj[s].mean()), 2) if s.any() else None,
                         "rejections": rej})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(f"{len(det.frame)} accepted detections in {len(np.unique(det.frame))} of {det.n_frames} frames. Accepted = detector gate "
                   "(pixel area >= 500, IPPE ambiguity ratio <= 0.7, IDs 0-9). Rejections are detector-level: unknown_id, too_small, ambiguous.")
        if not rej_known:
            st.warning("This cache was made before rejection counts were stored. Recompute to fill the rejections column (about 1-2 min).")
            if st.button("Recompute detections with rejection counts"):
                bar = st.progress(0.0)
                detect_run(RunDataset(run_dir), cache_path=OUT / f"{name}_detections.npz", force=True,
                           progress=lambda i, n: bar.progress(min(i / n, 1.0)))
                st.rerun()
        st.bar_chart(df.set_index("ID")["detections"])

        st.subheader("Overlay preview")
        st.caption("Frames are decoded from the video and the cached detections are drawn on them (marker outline, pose axes, ID and distance); "
                   "rejected detections are not drawn. Scrub with the slider or press Play.")
        K, dist, _ = load_calibration()
        n_frames = len(stt["frame_t"])
        by_frame: dict[int, list] = {}
        for f, m, rv, tv, d in zip(det.frame, det.marker_id, det.rvec, det.tvec, det.distance):
            by_frame.setdefault(int(f), []).append((int(m), rv, tv, float(d)))
        h = 0.15 / 2
        sq = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)

        def render(frame, f):
            for m, rv, tv, d in by_frame.get(f, []):
                pts, _ = cv2.projectPoints(sq, rv, tv, K, dist)
                cv2.polylines(frame, [pts.reshape(-1, 2).astype(np.int32)], True, (0, 220, 0), 3)
                cv2.drawFrameAxes(frame, K, dist, rv, tv, 0.075, 4)
                x, y = pts.reshape(-1, 2)[0].astype(int)
                cv2.putText(frame, f"id {m}  {d:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 3)
            cv2.putText(frame, f"frame {f}  t={(stt['frame_t'][min(f, n_frames - 1)] - stt['frame_t'][0]) * 1e-9:.2f}s", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
            w = 960
            frame = cv2.resize(frame, (w, int(frame.shape[0] * w / frame.shape[1])), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        first_det = int(det.frame.min()) if len(det.frame) else 0
        f0 = st.slider("Frame", 0, n_frames - 1, first_det, key=f"frame_{name}")
        c1, c2 = st.columns([1, 3])
        play = c1.button("Play 150 frames from here")
        stride = c2.select_slider("Playback stride (frames)", [1, 2, 3, 5], value=3)
        slot = st.empty()
        cap = cv2.VideoCapture(stt["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        ok, fr = cap.read()
        if ok:
            slot.image(render(fr, f0), width="stretch")
        else:
            st.warning("Could not decode that frame.")
        if play and ok:
            f = f0
            for _ in range(150 // stride):
                for _ in range(stride):
                    ok, fr = cap.read()
                    f += 1
                if not ok:
                    break
                slot.image(render(fr, f), width="stretch")
                time.sleep(0.03)
        cap.release()

# ---- tab 3 --------------------------------------------------------------------------------------------------
with tab_traj:
    st.subheader("Corrected trajectory over the marker map")
    rp = OUT / f"{name}_app_slam.npz"
    if not rp.is_file():
        st.info("No SLAM result for this run yet: press **Run detection + SLAM** in tab 1.")
    else:
        res = load_app_result(str(rp), rp.stat().st_mtime)
        used_map = str(res["map_name"])
        mmap = MarkerMap.load(OUT / used_map) if (OUT / used_map).is_file() else None
        if mmap is None or mmap.T_room_world is None:
            st.error(f"Map `{used_map}` used for this result is no longer available.")
        else:
            c = st.columns(3)
            show_m4 = c[0].checkbox("Overlay M4 (no-marker) trajectory", value=True)
            show_ell = c[1].checkbox("Show per-pose uncertainty ellipses (2-sigma)", value=True)
            show_gtm = c[2].checkbox("Show surveyed marker positions", value=True)
            markers_room = {m: (mmap.T_room_world @ T) for m, T in mmap.poses.items()}
            try:
                gt_path = parse_polyline(gt_text) if gt_text.strip() else None
            except Exception:
                gt_path = None
                st.warning("Could not parse the ground-truth path; expected `x,y;x,y;...`.")
            left, right = st.columns([3, 2])
            with left:
                fig = traj_figure(res, markers_room, load_ground_truth(GT_CSV), gt_path, show_m4, show_ell, show_gtm)
                st.pyplot(fig, clear_figure=True)
            with right:
                p, base, cov = res["p"], res["baseline_p"], res["cov"]
                st.markdown(f"**Map used:** `{used_map}`  \n**Keyframes:** {len(p)}  \n**Marker observations:** {int(res['n_obs'])} "
                            f"({int(res['dropped'])} dropped as outliers)  \n**Loop-closure edges:** {int(res['n_loops'])}")
                st.metric("Corrected path length", f"{path_len(p):.2f} m", f"M4: {path_len(base):.2f} m", delta_color="off")
                st.metric("Corrected end point", f"({p[-1, 0]:.2f}, {p[-1, 1]:.2f}) m", f"M4: ({base[-1, 0]:.2f}, {base[-1, 1]:.2f})", delta_color="off")
                if gt_path is not None:
                    from src.run_slam import polyline_error
                    e1, e0 = polyline_error(p[:, :2], gt_path), polyline_error(base[:, :2], gt_path)
                    st.metric("Mean distance to taped path", f"{100 * e1.mean():.0f} cm", f"M4 only: {100 * e0.mean():.0f} cm", delta_color="off")
                    st.metric("End-point error", f"{100 * np.linalg.norm(p[-1, :2] - gt_path[-1]):.0f} cm",
                              f"M4 only: {100 * np.linalg.norm(base[-1, :2] - gt_path[-1]):.0f} cm", delta_color="off")
                    if name in ("run_2", "run_5"):
                        st.caption("† Measured against the path you entered (run 1's taped L). This run only approximated that path: same start, but its turning and "
                                   "end points were re-walked rather than re-taped, and the deviation was not measured, so these numbers include unmeasured route deviation.")
                if cov.size:
                    sig = np.sqrt(np.trace(cov[:, 3:, 3:], axis1=1, axis2=2))
                    st.metric("Position 1-sigma (median / max)", f"{100 * np.median(sig):.0f} cm / {100 * sig.max():.0f} cm")
                st.caption("M4 is placed in the room frame by its first marker-posed keyframe only (yaw + translation, no scale correction). "
                           "Ellipses come from (J^T J)^-1 and ignore the robust loss: read them as relative uncertainty, large where no mapped marker is in view.")
                if used_map == "marker_map.json":
                    st.warning("This map pools all runs, including the selected one, so the result is optimistic.")

# ---- tab 4 --------------------------------------------------------------------------------------------------
with tab_diag:
    st.subheader("Saved results (M6)")
    st.caption("These were produced by the command-line scripts and are shown as saved.")
    d1, d2, d3 = st.tabs(["Baseline comparison", "Loop closure / relocalization / ablation", "Reruns: difficult condition & repeatability"])
    with d1:
        st.markdown("**Five configurations, all from run 1**")
        show_file(OUT / "baseline_table.md", "md", 'python -m src.run_baselines "data/all runs/run 1" --map outputs/marker_map_excl_run1.json --gt-path "2.9,3.25;0.5,3.25;0.5,0.25"')
        show_file(OUT / "baseline_table.png", "img", "python -m src.run_baselines ...")
    with d2:
        st.markdown("**Loop closure on run 3**")
        show_file(OUT / "loop_closure_run_3.md", "md", 'python -m src.run_loop_closure "data/all runs/run 3"')
        show_file(OUT / "loop_closure_run_3.png", "img", 'python -m src.run_loop_closure "data/all runs/run 3"')
        st.markdown("**Relocalization on run 4** (map built without run 4)")
        show_file(OUT / "relocalization_run_4.md", "md", 'python -m src.run_relocalization "data/all runs/run 4" --map outputs/marker_map_excl_run4.json')
        st.markdown("**Marker-density ablation (run 3)**")
        show_file(OUT / "ablation.png", "img", 'python -m src.run_ablation "data/all runs/run 3"')
        show_file(OUT / "ablation.md", "md", 'python -m src.run_ablation "data/all runs/run 3"')
    with d3:
        cmd = "python -m src.run_pair_eval"
        st.markdown("**Difficult condition: the out-and-back re-recording (`rerun 5`) vs run 1 (easy) and the original run 5**")
        st.warning("The system fails on rerun 5: only 6% of frames see a marker and the pose graph estimates under 1 m of an out-and-back path "
                   "that is 10.8 m taped. Its pose-graph distance/end-point numbers look small only because the estimate never leaves the start; "
                   "read the stress and detection rows, not the accuracy rows. Loop residual cannot be computed (no marker is seen twice).")
        st.caption("† = original run 5 (and run 2) only approximated run 1's taped path (turning/end points re-walked, not re-taped; deviation not measured), "
                   "so their taped-path rows include unmeasured route deviation. The rerun 5 vs run 1 result rests on image and detection measurements, not on that.")
        show_md_section(OUT / "rerun_evaluation.md", "Is the out-and-back rerun", cmd)
        st.markdown("**Repeatability across repeated walks of a nominally identical route**")
        st.info("The five one-way walks are only nominally the same route: run 1, rerun 1 and rerun 2 followed the taped path; original runs 2 and 5 were "
                "re-walked without re-taping, so part of the n = 5 spread reflects route variation, not estimator error alone. "
                "The n = 3 rows (true taped repeats only) are shown alongside so you can see how much that matters.")
        show_md_section(OUT / "rerun_evaluation.md", "Repeatability", cmd)
        show_file(OUT / "repeatability.png", "img", cmd)
        st.markdown("**Per-run paths (pose graph vs M4 vs taped path)**")
        st.caption("† run 2 and run 5 are scored against run 1's taped path, which they approximated but did not follow exactly.")
        show_file(OUT / "rerun_evaluation_paths.png", "img", cmd)
    st.markdown("**Marker loss (run 2)**")
    show_file(OUT / "run2_markerloss.md", "md", 'python -m src.run_marker_loss "data/all runs/run 2" --map outputs/marker_map_excl_run2.json')
    show_file(OUT / "run2_markerloss.png", "img", 'python -m src.run_marker_loss "data/all runs/run 2" --map outputs/marker_map_excl_run2.json')
    st.markdown("**Difficult condition (run 5 vs run 1)**")
    show_file(OUT / "run5_difficult.md", "md", 'python -m src.run_difficult "data/all runs/run 5" --baseline "data/all runs/run 1" --map outputs/marker_map_excl_run5.json --baseline-map outputs/marker_map_excl_run1.json')
    show_file(OUT / "run5_difficult.png", "img", "python -m src.run_difficult ...")
    show_file(OUT / "run5_difficult_paths.png", "img", "python -m src.run_difficult ...")
