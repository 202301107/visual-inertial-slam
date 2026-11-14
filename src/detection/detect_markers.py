"""ArUco detection plus camera-to-marker pose (solvePnP IPPE_SQUARE) with a rejection gate."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CALIB_PATH = Path(__file__).resolve().parents[2] / "calibration.npz"


@dataclass
class DetectorConfig:
    marker_length: float = 0.15          # black square side, metres
    dictionary: str = "DICT_4X4_50"
    min_area_px: float = 500.0           # reject if the quad's pixel area is below this
    # IPPE returns two candidate poses. Reject if best_err / second_err exceeds this
    # (1.0 = both fit equally well = orientation flip cannot be resolved).
    max_ambiguity_ratio: float = 0.7
    # Real marker set (data/markers.csv). Any other detected ID is dropped as "unknown_id".
    valid_ids: frozenset[int] = frozenset(range(10))


@dataclass
class MarkerDetection:
    id: int
    corners: np.ndarray       # (4, 2) pixels, order TL, TR, BR, BL
    rvec: np.ndarray          # (3,) Rodrigues, marker frame -> camera frame
    tvec: np.ndarray          # (3,) marker origin in camera frame, metres
    distance: float           # |tvec|, metres
    area_px: float
    reproj_err: float         # RMS pixel error of the chosen pose
    ambiguity_ratio: float    # best_err / second_err, lower is less ambiguous


@dataclass
class RejectedMarker:
    id: int
    corners: np.ndarray
    reason: str               # "unknown_id", "too_small", "ambiguous" or "behind_camera"


@dataclass
class FrameDetections:
    detections: list[MarkerDetection] = field(default_factory=list)
    rejected: list[RejectedMarker] = field(default_factory=list)

    @property
    def ids(self) -> np.ndarray:
        return np.array([d.id for d in self.detections], dtype=int)

    @property
    def rvecs(self) -> np.ndarray:
        return np.array([d.rvec for d in self.detections]).reshape(-1, 3)

    @property
    def tvecs(self) -> np.ndarray:
        return np.array([d.tvec for d in self.detections]).reshape(-1, 3)

    @property
    def corners(self) -> np.ndarray:
        return np.array([d.corners for d in self.detections]).reshape(-1, 4, 2)


def load_calibration(path: str | Path = DEFAULT_CALIB_PATH):
    """Return (K, dist, (width, height)) from calibration.npz."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    d = np.load(path)
    return d["K"].astype(np.float64), d["dist"].astype(np.float64), tuple(int(v) for v in d["image_size"])


class MarkerDetector:
    def __init__(self, K: np.ndarray, dist: np.ndarray, config: DetectorConfig | None = None,
                 image_size: tuple[int, int] | None = None):
        self.K = K
        self.dist = dist
        self.cfg = config or DetectorConfig()
        self.image_size = image_size   # (w, h) the intrinsics were calibrated at
        self._size_checked = False

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.cfg.dictionary))
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

        h = self.cfg.marker_length / 2.0
        self._obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    def detect(self, frame: np.ndarray) -> FrameDetections:
        if not self._size_checked and self.image_size is not None:
            self._size_checked = True
            if (frame.shape[1], frame.shape[0]) != tuple(self.image_size):
                warnings.warn(
                    f"Frame size {frame.shape[1]}x{frame.shape[0]} differs from calibration size "
                    f"{self.image_size[0]}x{self.image_size[1]}; poses will be wrong "
                    f"unless the camera was calibrated at this resolution/orientation.")

        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self._detector.detectMarkers(gray)
        out = FrameDetections()
        if ids is None:
            return out

        for corners, marker_id in zip(corners_list, ids.ravel()):
            c = corners.reshape(4, 2).astype(np.float32)
            if int(marker_id) not in self.cfg.valid_ids:
                out.rejected.append(RejectedMarker(int(marker_id), c, "unknown_id"))
                continue
            area = float(cv2.contourArea(c))
            if area < self.cfg.min_area_px:
                out.rejected.append(RejectedMarker(int(marker_id), c, "too_small"))
                continue

            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                self._obj, c, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if n == 0:
                out.rejected.append(RejectedMarker(int(marker_id), c, "ambiguous"))
                continue
            errs = np.asarray(errs, dtype=np.float64).ravel()
            order = np.argsort(errs)          # best solution first
            best = order[0]
            if len(order) > 1:
                second_err = errs[order[1]]
                ratio = errs[best] / second_err if second_err > 1e-9 else 1.0
            else:
                ratio = 0.0
            if ratio > self.cfg.max_ambiguity_ratio:
                out.rejected.append(RejectedMarker(int(marker_id), c, "ambiguous"))
                continue

            rvec = rvecs[best].reshape(3)
            tvec = tvecs[best].reshape(3)
            if tvec[2] <= 0:
                out.rejected.append(RejectedMarker(int(marker_id), c, "behind_camera"))
                continue
            out.detections.append(MarkerDetection(
                id=int(marker_id), corners=c, rvec=rvec, tvec=tvec,
                distance=float(np.linalg.norm(tvec)), area_px=area,
                reproj_err=float(errs[best]), ambiguity_ratio=float(ratio)))
        return out
