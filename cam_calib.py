"""Calibrate the phone camera from a checkerboard VIDEO recorded in OpenCamera Sensors.
Usage: python calibrate_from_video.py calib_video.mp4 [--square 0.025]
Board: 9x6 INNER corners (10x7 squares). Saves calibration.npz (K, dist, image_size)."""
import cv2, numpy as np, sys, argparse

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--square", type=float, default=0.025, help="square side in metres (measure it!)")
p.add_argument("--cols", type=int, default=9)   # inner corners
p.add_argument("--rows", type=int, default=6)
p.add_argument("--step", type=int, default=10, help="test every Nth frame")
p.add_argument("--max", type=int, default=40, help="max images to use")
a = p.parse_args()
pattern = (a.cols, a.rows)

objp = np.zeros((a.cols * a.rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square

cap = cv2.VideoCapture(a.video)          # OpenCV applies the rotation flag automatically
objpoints, imgpoints, size, centers = [], [], None, []
i = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    i += 1
    if i % a.step: continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]               # (width, height) as read
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 20:   # skip blurry frames
        continue
    ok, corners = cv2.findChessboardCorners(gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok: continue
    c = corners.reshape(-1, 2).mean(0)
    # keep only frames whose board position differs from ones already kept (variety)
    if any(np.linalg.norm(c - q) < 15 for q in centers): continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    objpoints.append(objp); imgpoints.append(corners); centers.append(c)
    if len(objpoints) >= a.max: break

print("frame size (w,h):", size, "| good images:", len(objpoints))
if len(objpoints) < 12:
    sys.exit("Fewer than 12 good images - film the board again: slower, brighter light, more tilts/positions.")
rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
print("reprojection error (px):", round(rms, 3))
print("fx, fy:", K[0, 0], K[1, 1], " cx, cy:", K[0, 2], K[1, 2])
np.savez("calibration.npz", K=K, dist=dist, image_size=np.array(size))
print("saved calibration.npz")