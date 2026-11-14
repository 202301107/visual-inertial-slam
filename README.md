# Visual–Inertial SLAM with Fiducial Landmark Anchors

IE415 Control of Autonomous Systems — course project.

Indoor mapping and localization from phone camera + IMU, anchored by fixed ArUco markers.
A person walks through a marked room holding a phone; the system reconstructs the marker
map and the walked trajectory, and corrects drift whenever a marker is re-observed.

## Architecture
Architecture C (Phone–Laptop Hybrid), processed offline.
Phone logs video + IMU + timestamps (OpenCamera Sensors) → laptop runs the full Python pipeline.

## Stack
- Python 3.14 venv (`venv/`), run everything from the repository root
- OpenCV (`opencv-contrib-python`) — ArUco detection (DICT_4X4_50, 15 cm markers), calibration, ORB visual odometry
- NumPy / SciPy — math, IMU preintegration, pose-graph optimisation with `scipy.optimize.least_squares`
  (GTSAM was dropped: it publishes no Windows wheels)
- Matplotlib — plots
- Streamlit — interface

## Setup
```bash
git clone <repo-url>
cd visual-inertial-slam
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt
```
`data/`, `outputs/`, `venv/`, `*.mp4` and `*.npz` are git-ignored: recordings, calibration and every generated result live
only on the machine that produced them.

## Pipeline
`acquisition → detection → odometry + IMU → estimation (pose graph) → mapping → interface`

| Stage | Module | What it does |
|---|---|---|
| Acquisition | `src/acquisition/dataset_loader.py` | Finds video, accel, gyro, magnetic and frame-timestamp files by pattern anywhere under a run folder (they may sit in different subfolders); yields frames with nanosecond timestamps; `imu_between(t0, t1)` |
| Calibration | `cam_calib.py` | Intrinsics from a checkerboard video → `calibration.npz` |
| Detection | `src/detection/detect_markers.py`, `detect_run.py` | ArUco + `solvePnP` IPPE_SQUARE pose; gates on pixel area, IPPE ambiguity and an ID whitelist (0–9); per-run cache `outputs/<run>_detections.npz` (accepted detections + rejection counts) |
| Odometry (M4) | `src/odometry/` | Gyro orientation (static-start bias, gravity tilt correction); VO translation direction re-estimated with the gyro rotation held fixed (essential-matrix rotation is unreliable at these baselines); IMU speed along that direction with zero-velocity updates and a velocity leak. **Metric scale is unreliable.** Modes `vio`, `vo`, `imu` |
| Estimation | `src/estimation/pose_graph.py` | Least-squares graph over keyframe poses (SO(3) exponential-map perturbations): odometry factors with a free scale per time window, marker factors, optional loop-closure relative-pose edges, optional marker priors; covariance from (JᵀJ)⁻¹ |
| Mapping | `src/mapping/marker_map.py` | Marker poses in a reference-marker frame from one or many runs; odometry bridges markers that are never co-visible; optional `data/markers.csv` priors; saved to `outputs/marker_map*.json` with a fit to the surveyed room frame |
| Interface | `src/interface/app.py` | Streamlit app (below) |

Acquisition and estimation are kept separate (the run folder is the boundary) so one recording can be replayed while
the algorithm improves.

## Data
```
data/
  markers.csv        surveyed marker positions + facing (room frame); ids 0-6 are map markers, 7-9 floor scoring points
  room.md            coordinate frame, start tape (2.9, 3.25), run 1 taped L path, notes
  recordings/        calibration video
  all runs/          run 1-5, rerun 1, rerun 2, rerun 5   (video at run level, CSVs in a timestamped subfolder, or all together)
```
Room frame: origin at the S/E wall corner, X toward N, Y toward W, Z up. Taped path (run 1, rerun 1, rerun 2):
(2.9, 3.25) → (0.5, 3.25) → (0.5, 0.25), 5.4 m. Rerun 5 is the same L walked out and back (10.8 m, ends at the start tape).
Original runs 2 and 5 are one-way and roughly L-shaped with the same start, but their turning/end points were re-walked,
not re-taped; any taped-path error for them is measured against run 1's path and marked †.

## Running it
```bash
python cam_calib.py data/recordings/calib_video.mp4          # once -> calibration.npz
python -m src.sanity_check "data/all runs/run 1"             # overlay video + per-marker table
python -m src.detection.detect_run "data/all runs/run 1"     # cache detections (+ rejections)
python -m src.run_trajectory "data/all runs/run 1"           # M4 baseline -> outputs/run_1_traj.csv/.png
python -m src.mapping.marker_map "data/all runs/run 1" "data/all runs/run 2" ...  --out outputs/marker_map.json [--use-priors]
python -m src.run_slam "data/all runs/run 1" --map outputs/marker_map_excl_run1.json --gt-path "2.9,3.25;0.5,3.25;0.5,0.25"
streamlit run src/interface/app.py
```
The mapping CLI needs a detections cache and an M4 trajectory for every run it pools. Pass explicit run folders to build
leave-one-out maps (`marker_map_excl_runN.json`: without run N; `marker_map_excl_pairN.json`: without a run and its re-recording).

Evaluation scripts (all reuse cached detections and trajectories):

| Script | Output |
|---|---|
| `src/run_loop_closure.py` (run 3) | `loop_closure_run_3.md/.png` |
| `src/run_relocalization.py` (run 4, cold start) | `relocalization_run_4.md` |
| `src/run_baselines.py` (run 1, five configurations) | `baseline_table.md/.png` |
| `src/run_ablation.py` (run 3, marker density) | `ablation.md/.png` |
| `src/run_marker_loss.py` (run 2 / rerun 2) | `run2_markerloss*.md/.png`, `rerun2_markerloss.md/.png` |
| `src/run_difficult.py` (run 5 vs run 1) | `run5_difficult.md/.png` |
| `src/run_pair_eval.py` (originals vs reruns, repeatability) | `rerun_evaluation.md`, `rerun_evaluation_paths.png`, `repeatability.png` |

## Interface
`streamlit run src/interface/app.py` — pick a run under `data/all runs`; see camera/IMU status; run detection + SLAM (cached
detections and the M4 trajectory are reused; the saved-map selector defaults to a map built without the selected run);
inspect the per-marker table (detections, distance, reprojection error, rejections) and a frame-by-frame overlay preview; view
the corrected trajectory over the marker map with the M4 no-marker path and toggleable uncertainty ellipses; and browse the
saved results (baseline table, loop closure, relocalization, ablation, marker loss, difficult condition, repeatability).

## Results at a glance
All numbers are against maps built without the evaluated run unless stated; details and caveats are in the files named above.
- **Marker map:** wall markers about 20–40 cm RMS against the survey (24 cm from run 3 alone, 25 cm pooled with priors, 36 cm pooled without).
  Directly co-visible markers agree to a few cm; markers linked only by odometry are the weak point (the walker crosses gaps 4–10× faster than M4's speed model assumes).
- **Loop closure (run 3):** the two estimates of marker 0 end up 0.79 m / 6.9° apart without it and 0.01 m / 0.6° with it; map RMS 37 → 24 cm.
- **Relocalization (run 4):** first accepted fix 4.6 s after the start of replay, 12 cm from the surveyed-marker pose.
- **Baselines (run 1, endpoint drift):** IMU only ≈ 1.3 km, visual odometry 3.4 m, visual-inertial 4.0 m, + markers 0.55 m, + loop closure 0.55 m
  (no gain: fixed map, one revisit pair). The 0.3–0.5 m phone-ahead-of-body offset is included in every endpoint figure.
- **Marker density (run 3):** mean error stays flat from 10 down to 6 markers, then rises; with none it is 1.26 m.
- **Marker loss (run 2):** a 9.1 s gap with no landmark; position 1σ grows from about 4 cm to about 3 m (conservative); the correction at re-acquisition is 84 cm (rerun 2: 26 cm).
- **Difficult condition:** original run 5 is dim but not hard (detection did not degrade). Rerun 5 is meaningfully harder on all five stress criteria
  (dim, 99°/s angular rate, blurrier, 6% of frames see a marker) and the system fails on it: the estimate collapses to under 1 m of a 10.8 m path.
  Use it as a limits case, not an accuracy result.
- **Reruns vs originals:** keep the originals for runs 1 and 2 (rerun 1 is less accurate; rerun 2's apparent gain is within walk-to-walk scatter and is
  overstated by the unmeasured route deviation of original run 2).
- **Repeatability (nominally identical L route):** end-point error 103 ± 47 cm over n = 5 walks, 76 ± 37 cm over the n = 3 true taped repeats;
  part of the n = 5 spread is route variation (runs 2 and 5 were re-walked, not re-taped).

## Assumptions and limitations
- Camera–IMU time offset is assumed zero (measured as 0 to +40 ms on run 1); accelerometer bias and lever arm are not modelled.
- Metric scale of the visual-inertial odometry is unreliable; the pose graph solves a per-window scale and a map or markers supply the metric information.
- The markers are mounted rotated 180° in-plane (their up axis points down); `data/markers.csv` records only position and facing, so the in-plane rotation used for surveyed truth comes from the data.
- Taped ground truth exists for run 1 and the reruns only; runs 2–5 otherwise use survey-derived reference positions (the camera position implied by the surveyed pose of a wall marker in view), which carry the map's own error.
- The saved `marker_map.json` pools runs 1–5; the three reruns were added after it was built and are not in it.

## Status (milestone checklist)
- [x] M1 — Proposal, architecture, stack, system diagram
- [x] M2 — Recording, timestamps, camera calibration
- [x] M3 — ArUco detection + camera-to-marker pose
- [x] M4 — Trajectory estimation without marker correction
- [x] M5 — Marker map + trajectory visualization
- [x] M6 — Loop closure, relocalization, baseline comparison
- [ ] M7 — Final experiments and interface done; report and demo remain

## Repository layout
```
cam_calib.py, generate_markers.py      calibration, marker image generator
src/acquisition, detection, odometry, estimation, mapping, evaluation, interface   pipeline stages (see table above)
src/geometry.py                        shared SE(3) helpers
src/run_*.py, src/sanity_check.py      command-line entry points
data/                                  recordings and survey (git-ignored)
outputs/                               generated results, caches, maps (git-ignored)
```
