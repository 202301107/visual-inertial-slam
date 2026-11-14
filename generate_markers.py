import cv2
import os

# Settings
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size_px = 600      # resolution of the saved image (not physical size)
num_markers = 10          # IDs 0 to 9
output_dir = "markers"

os.makedirs(output_dir, exist_ok=True)

for marker_id in range(num_markers):
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
    filename = os.path.join(output_dir, f"marker_{marker_id}.png")
    cv2.imwrite(filename, img)
    print(f"Saved {filename}")