"""Run marker detection over a whole run once and cache the result.

The cache is an .npz of per-detection rows (frame, timestamp, marker id, rvec, tvec, area,
reprojection error), so mapping and SLAM can reuse it without re-decoding the video.
rvec/tvec are marker -> camera (x_cam = R x_marker + t), as returned by detect_markers.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_markers import (DEFAULT_CALIB_PATH, DetectorConfig, MarkerDetector,
                                          load_calibration)


@dataclass
class RunDetections:
    frame: np.ndarray      # (M,) frame index
    t_ns: np.ndarray       # (M,)
    marker_id: np.ndarray  # (M,)
    rvec: np.ndarray       # (M, 3)
    tvec: np.ndarray       # (M, 3)
    area: np.ndarray       # (M,) pixel area of the detected quad
    reproj: np.ndarray     # (M,) IPPE reprojection error, px
    n_frames: int
    rej_marker: np.ndarray | None = None   # (R,) id of each detector-rejected marker (None in old caches)
    rej_reason: np.ndarray | None = None   # (R,) 'unknown_id' | 'too_small' | 'ambiguous' | 'behind_camera'

    @property
    def distance(self) -> np.ndarray:
        return np.linalg.norm(self.tvec, axis=1)

    def select(self, mask: np.ndarray) -> "RunDetections":
        return RunDetections(self.frame[mask], self.t_ns[mask], self.marker_id[mask],
                             self.rvec[mask], self.tvec[mask], self.area[mask], self.reproj[mask],
                             self.n_frames, self.rej_marker, self.rej_reason)

    def save(self, path) -> None:
        extra = {} if self.rej_marker is None else {"rej_marker": self.rej_marker, "rej_reason": self.rej_reason}
        np.savez(path, frame=self.frame, t_ns=self.t_ns, marker_id=self.marker_id, rvec=self.rvec,
                 tvec=self.tvec, area=self.area, reproj=self.reproj, n_frames=self.n_frames, **extra)

    @staticmethod
    def load(path) -> "RunDetections":
        d = np.load(path)
        has = "rej_marker" in d.files
        return RunDetections(d["frame"], d["t_ns"], d["marker_id"], d["rvec"], d["tvec"],
                             d["area"], d["reproj"], int(d["n_frames"]),
                             d["rej_marker"] if has else None, d["rej_reason"] if has else None)


def detect_run(ds: RunDataset, calib_path=DEFAULT_CALIB_PATH, config: DetectorConfig | None = None,
               cache_path: str | Path | None = None, force: bool = False, progress=None) -> RunDetections:
    """Detections for every frame of a run (loaded from cache_path if it exists)."""
    if cache_path is not None and Path(cache_path).is_file() and not force:
        return RunDetections.load(cache_path)
    K, dist, size = load_calibration(calib_path)
    det = MarkerDetector(K, dist, config, image_size=size)
    rows, rej = [], []
    for idx, t_ns, frame in ds.frames():
        res = det.detect(frame)
        for d in res.detections:
            rows.append((idx, t_ns, d.id, *d.rvec, *d.tvec, d.area_px, d.reproj_err))
        rej += [(r.id, r.reason) for r in res.rejected]
        if progress is not None:
            progress(idx, len(ds.frame_t_ns))
        if idx % 200 == 0:
            print(f"  detect frame {idx}/{len(ds.frame_t_ns)}", file=sys.stderr)
    a = np.array(rows, dtype=np.float64).reshape(-1, 11)
    out = RunDetections(a[:, 0].astype(int), np.array([r[1] for r in rows], dtype=np.int64),
                        a[:, 2].astype(int), a[:, 3:6], a[:, 6:9], a[:, 9], a[:, 10], len(ds.frame_t_ns),
                        np.array([r[0] for r in rej], dtype=int), np.array([r[1] for r in rej], dtype="<U16"))
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        out.save(cache_path)
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Detect markers over a whole run and cache the result.")
    ap.add_argument("run_dir")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    name = Path(a.run_dir).resolve().name.strip().replace(" ", "_")
    path = Path(a.out_dir) / f"{name}_detections.npz"
    r = detect_run(RunDataset(a.run_dir), cache_path=path, force=a.force)
    ids, counts = np.unique(r.marker_id, return_counts=True)
    print(f"{len(r.frame)} detections over {len(np.unique(r.frame))}/{r.n_frames} frames -> {path}")
    print("per-ID counts:", dict(zip(ids.tolist(), counts.tolist())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
