# Visual–Inertial SLAM with Fiducial Landmark Anchors

IE415 Control of Autonomous Systems — course project.

Indoor mapping and localization from phone camera + IMU, anchored by fixed ArUco markers.
A person walks through a marked room holding a phone; the system reconstructs the marker
map and the walked trajectory, and corrects drift whenever a marker is re-observed.

## Architecture
Architecture C (Phone–Laptop Hybrid), processed offline.
Phone logs video + IMU + timestamps → laptop runs the full Python pipeline.

## Stack
- Python 3.10+
- OpenCV (`opencv-contrib-python`) — ArUco detection, calibration, ORB visual odometry
- GTSAM — pose-graph optimization (EKF fallback)
- NumPy / SciPy — math, IMU preintegration
- Matplotlib — plots
- Streamlit — interface

## Setup
```bash
git clone <repo-url>
cd visual-inertial-slam
pip install -r requirements.txt
```

## Pipeline
`acquisition → detection → odometry + IMU → estimation (graph/EKF) → mapping → interface`
Acquisition and estimation are kept separate (dataset folder is the boundary) so one
recording can be replayed while the algorithm improves.

## Status (milestone checklist)
- [x] M1 — Proposal, architecture, stack, system diagram
- [ ] M2 — Recording, timestamps, camera calibration
- [ ] M3 — ArUco detection + camera-to-marker pose
- [ ] M4 — Trajectory estimation without marker correction
- [ ] M5 — Marker map + trajectory visualization
- [ ] M6 — Loop closure, relocalization, baseline comparison
- [ ] M7 — Final experiments, interface, report, demo

## Repository layout
See `src/` for pipeline stages, `config/` for calibration and marker setup sheet,
`docs/` for the proposal and system diagram.
