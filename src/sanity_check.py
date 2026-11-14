"""Run detection over a recorded run, write an overlay video and print per-marker stats.

    python -m src.sanity_check "path/to/run 5"
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.acquisition.dataset_loader import RunDataset
from src.detection.detect_markers import (DEFAULT_CALIB_PATH, DetectorConfig, MarkerDetector,
                                          load_calibration)

GREEN, RED = (0, 220, 0), (0, 0, 255)


def parse_ids(text: str) -> frozenset[int]:
    """'0-9' or '0-6,12' -> set of ints."""
    ids = set()
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        ids.update(range(int(lo), int(hi or lo) + 1))
    return frozenset(ids)


def draw(frame, result, K, dist, marker_length, header):
    for r in result.rejected:
        cv2.polylines(frame, [r.corners.astype(np.int32)], True, RED, 2)
        x, y = r.corners[0].astype(int)
        cv2.putText(frame, f"{r.id} {r.reason}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    for d in result.detections:
        cv2.polylines(frame, [d.corners.astype(np.int32)], True, GREEN, 2)
        cv2.drawFrameAxes(frame, K, dist, d.rvec, d.tvec, marker_length * 0.5, 3)
        x, y = d.corners[0].astype(int)
        cv2.putText(frame, f"id {d.id}  {d.distance:.2f} m", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(frame, header, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(frame, header, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
    return frame


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir", help="run folder (contains the session subfolder)")
    ap.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    ap.add_argument("--out", default=None, help="overlay video path (default outputs/<run>_overlay.mp4)")
    ap.add_argument("--no-video", action="store_true", help="print stats only")
    ap.add_argument("--show", action="store_true", help="live preview window (q to quit)")
    ap.add_argument("--scale", type=float, default=0.5, help="overlay video scale (default 0.5)")
    ap.add_argument("--step", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--marker-length", type=float, default=DetectorConfig.marker_length)
    ap.add_argument("--min-area", type=float, default=DetectorConfig.min_area_px)
    ap.add_argument("--max-ambiguity", type=float, default=DetectorConfig.max_ambiguity_ratio)
    ap.add_argument("--valid-ids", type=parse_ids, default=DetectorConfig().valid_ids,
                    help="accepted ArUco IDs, e.g. 0-9 or 0-6,12 (default 0-9)")
    a = ap.parse_args(argv)

    ds = RunDataset(a.run_dir)
    print(ds.summary())

    K, dist, calib_size = load_calibration(a.calib)
    cfg = DetectorConfig(marker_length=a.marker_length, min_area_px=a.min_area,
                         max_ambiguity_ratio=a.max_ambiguity, valid_ids=a.valid_ids)
    det = MarkerDetector(K, dist, cfg, image_size=calib_size)

    writer = None
    out_path = None
    if not a.no_video:
        out_path = Path(a.out) if a.out else Path("outputs") / f"{Path(a.run_dir).name.strip()}_overlay.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

    dists = defaultdict(list)
    errs = defaultdict(list)
    n_rejected = defaultdict(lambda: defaultdict(int))
    frames_done = frames_with_det = 0
    t0 = int(ds.frame_t_ns[0])
    stop = None if a.max_frames is None else a.max_frames * a.step

    try:
        for idx, t_ns, frame in ds.frames(step=a.step, stop=stop):
            res = det.detect(frame)
            frames_done += 1
            frames_with_det += bool(res.detections)
            for d in res.detections:
                dists[d.id].append(d.distance)
                errs[d.id].append(d.reproj_err)
            for r in res.rejected:
                n_rejected[r.id][r.reason] += 1

            if out_path is not None or a.show:
                vis = draw(frame, res, K, dist, cfg.marker_length,
                           f"frame {idx}  t={(t_ns - t0) / 1e9:.2f}s")
                if a.scale != 1.0:
                    vis = cv2.resize(vis, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
                if out_path is not None:
                    if writer is None:
                        fps = 1e9 / float(np.median(np.diff(ds.frame_t_ns))) / a.step
                        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps, (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if a.show:
                    cv2.imshow("sanity_check", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if frames_done % 200 == 0:
                print(f"  ...{frames_done} frames", file=sys.stderr)
    finally:
        if writer is not None:
            writer.release()
        if a.show:
            cv2.destroyAllWindows()

    print(f"\nframes processed: {frames_done}, with >=1 accepted marker: {frames_with_det} "
          f"({100 * frames_with_det / max(frames_done, 1):.0f}%)")
    print(f"gate: valid_ids={sorted(cfg.valid_ids)}, min_area={cfg.min_area_px:g}px^2, "
          f"max_ambiguity_ratio={cfg.max_ambiguity_ratio:g}\n")
    print(f"{'ID':>3} {'detections':>10} {'mean dist (m)':>14} {'min..max (m)':>14} "
          f"{'mean reproj px':>15}  rejected")
    for mid in sorted(set(dists) | set(n_rejected)):
        rej = ", ".join(f"{k}={v}" for k, v in sorted(n_rejected[mid].items())) or "-"
        if dists[mid]:
            dd = np.array(dists[mid])
            print(f"{mid:>3} {len(dd):>10} {dd.mean():>14.2f} {f'{dd.min():.2f}..{dd.max():.2f}':>14} "
                  f"{np.mean(errs[mid]):>15.2f}  {rej}")
        else:
            print(f"{mid:>3} {0:>10} {'-':>14} {'-':>14} {'-':>15}  {rej}")
    if not (set(dists) | set(n_rejected)):
        print("  (no markers detected)")
    if out_path is not None:
        print(f"\noverlay video: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
