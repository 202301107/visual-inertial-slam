"""Load one recorded run: video frames with timestamps plus accel/gyro streams.

A run folder holds VID_<ts>.mp4, VID_<ts>_imu_timestamps.csv, VID_<ts>accel.csv, VID_<ts>gyro.csv
and VID_<ts>magnetic.csv, possibly spread over different subfolders. Each is found by
recursive glob, never by name.
All timestamps are integer nanoseconds on one shared clock.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VIDEO_PATTERN = "VID_*.mp4"
ACCEL_PATTERN = "*accel.csv"
GYRO_PATTERN = "*gyro.csv"
FRAME_TS_PATTERN = "*_imu_timestamps.csv"
MAGNETIC_PATTERN = "*magnetic.csv"


@dataclass
class ImuData:
    """Timestamped 3-axis samples: t_ns is (N,) int64, xyz is (N, 3) float64."""
    t_ns: np.ndarray
    xyz: np.ndarray

    def __len__(self) -> int:
        return len(self.t_ns)

    def between(self, t0_ns: int, t1_ns: int) -> "ImuData":
        """Samples with t0_ns <= t < t1_ns."""
        lo = np.searchsorted(self.t_ns, t0_ns, side="left")
        hi = np.searchsorted(self.t_ns, t1_ns, side="left")
        return ImuData(self.t_ns[lo:hi], self.xyz[lo:hi])


def _parse_int_ns(col: np.ndarray) -> np.ndarray:
    # Parse as integers first so nanosecond epoch values keep full precision.
    try:
        return col.astype(np.int64)
    except ValueError:
        return col.astype(np.float64).astype(np.int64)


def _read_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([c.strip() for c in line.replace(";", ",").split(",")])
    return rows


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_frame_timestamps(path: Path) -> np.ndarray:
    """One timestamp (ns) per video frame, in order. Header line, if any, is skipped."""
    rows = [r for r in _read_rows(path) if _is_number(r[0])]
    if not rows:
        raise ValueError(f"No timestamps found in {path}")
    return _parse_int_ns(np.array([r[0] for r in rows]))


def load_imu_csv(path: Path) -> ImuData:
    """Headerless CSV with columns x, y, z, timestamp_ns. Returned sorted by time."""
    rows = [r for r in _read_rows(path) if len(r) >= 4 and all(_is_number(c) for c in r[:4])]
    if not rows:
        raise ValueError(f"No IMU rows found in {path}")
    arr = np.array([r[:4] for r in rows])
    t = _parse_int_ns(arr[:, 3])
    xyz = arr[:, :3].astype(np.float64)
    order = np.argsort(t, kind="stable")
    return ImuData(t[order], xyz[order])


def _find_one(root: Path, pattern: str, what: str) -> Path:
    """The single file under `root` (searched recursively) matching `pattern`."""
    hits = sorted(p for p in root.rglob(pattern) if p.is_file())
    if len(hits) != 1:
        found = "none" if not hits else "; ".join(str(p.relative_to(root)) for p in hits)
        raise FileNotFoundError(
            f"Expected exactly one {what} ({pattern}) under {root}; found {len(hits)}: {found}")
    return hits[0]


class RunDataset:
    """One run: video + frame timestamps + accel + gyro."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"Run folder not found: {self.run_dir}")
        # Each file is searched for independently: the video and the sensor CSVs may sit
        # in different folders of the run tree.
        self.video_path = _find_one(self.run_dir, VIDEO_PATTERN, "video")
        self.frame_ts_path = _find_one(self.run_dir, FRAME_TS_PATTERN, "frame timestamp file")
        self.accel_path = _find_one(self.run_dir, ACCEL_PATTERN, "accel file")
        self.gyro_path = _find_one(self.run_dir, GYRO_PATTERN, "gyro file")
        self.magnetic_path = _find_one(self.run_dir, MAGNETIC_PATTERN, "magnetic file")  # not loaded yet

        self.frame_t_ns = load_frame_timestamps(self.frame_ts_path)
        self.accel = load_imu_csv(self.accel_path)
        self.gyro = load_imu_csv(self.gyro_path)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video {self.video_path}")
        self.n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.image_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # (w, h)
        cap.release()

        if self.n_video_frames != len(self.frame_t_ns):
            warnings.warn(
                f"{self.video_path.name}: video reports {self.n_video_frames} frames but "
                f"{len(self.frame_t_ns)} timestamps; frames beyond the shorter count are dropped.")
        if np.any(np.diff(self.frame_t_ns) <= 0):
            warnings.warn("Frame timestamps are not strictly increasing.")

    def frames(self, start: int = 0, stop: int | None = None,
               step: int = 1) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield (frame_index, timestamp_ns, BGR frame). Skipped frames are still decoded."""
        n = len(self.frame_t_ns)
        stop = n if stop is None else min(stop, n)
        cap = cv2.VideoCapture(str(self.video_path))  # applies the rotation flag itself
        try:
            i = 0
            while i < stop:
                ok, frame = cap.read()
                if not ok:
                    if i < stop:
                        warnings.warn(f"Video ended at frame {i}, expected {stop}.")
                    break
                if i >= start and (i - start) % step == 0:
                    yield i, int(self.frame_t_ns[i]), frame
                i += 1
        finally:
            cap.release()

    def imu_between(self, t0_ns: int, t1_ns: int) -> tuple[ImuData, ImuData]:
        """(accel, gyro) samples with t0_ns <= t < t1_ns, e.g. between two frame timestamps."""
        return self.accel.between(t0_ns, t1_ns), self.gyro.between(t0_ns, t1_ns)

    def summary(self) -> str:
        def rate(t: np.ndarray) -> float:
            return 1e9 / float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
        t0 = self.frame_t_ns[0]
        return "\n".join([
            f"video:      {self.video_path.relative_to(self.run_dir)}  "
            f"{self.image_size[0]}x{self.image_size[1]}, "
            f"{len(self.frame_t_ns)} timestamps, ~{rate(self.frame_t_ns):.1f} fps",
            f"csv files:  {self.accel_path.parent.relative_to(self.run_dir)}",
            f"accel:      {len(self.accel)} samples, ~{rate(self.accel.t_ns):.0f} Hz",
            f"gyro:       {len(self.gyro)} samples, ~{rate(self.gyro.t_ns):.0f} Hz",
            f"video span: {(self.frame_t_ns[-1] - t0) / 1e9:.2f} s | IMU offset from first frame: "
            f"accel {(self.accel.t_ns[0] - t0) / 1e9:+.3f} s .. {(self.accel.t_ns[-1] - t0) / 1e9:+.3f} s, "
            f"gyro {(self.gyro.t_ns[0] - t0) / 1e9:+.3f} s .. {(self.gyro.t_ns[-1] - t0) / 1e9:+.3f} s",
        ])
