import cv2
import numpy as np
from time import time, sleep
from collections import deque
from picamera2 import Picamera2, Preview
from utils import compute_tsp_with_convex_hull, draw_path_on_image, detect_aruco_markers

# === CONFIG ===
CORNER_IDS = [1, 2, 3, 4]
MOVING_ID = 10
MAX_TRAIL = 100


def warp_perspective_with_corners(image, marker_map):
    centers = []
    for cid in CORNER_IDS:
        pts = marker_map[cid][0]
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        centers.append((cx, cy))

    centers = sorted(centers, key=lambda p: (p[1], p[0]))
    top = sorted(centers[:2], key=lambda p: p[0])
    bottom = sorted(centers[2:], key=lambda p: p[0])
    ordered = np.array([top[0], top[1], bottom[0], bottom[1]], dtype="float32")
    dst = np.array([[0, 0], [500, 0], [0, 500], [500, 500]], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (500, 500))
    return warped, M


def get_tsp_points(marker_map):
    tsp_points = []
    for id, corners in marker_map.items():
        if id in CORNER_IDS or id == MOVING_ID:
            continue
        pts = corners[0]
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        tsp_points.append((cx, cy))
    return tsp_points


def draw_trail(image, trail):
    for i in range(1, len(trail)):
        cv2.line(image, trail[i-1], trail[i], (0, 0, 255), 2)
    if trail:
        cv2.circle(image, trail[-1], 5, (0, 0, 255), -1)


def overlay_text(image, text, position=(10, 20)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def main():
    picam2 = Picamera2()
    picam2.start_preview(Preview.QTGL)
    config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    sleep(1)

    last_marker_ids = set()
    trail = deque(maxlen=MAX_TRAIL)

    prev_time = time()

    while True:
        frame = picam2.capture_array("main")
        marker_map, _ = detect_aruco_markers(frame)

        if not all(cid in marker_map for cid in CORNER_IDS):
            overlay_text(frame, "Waiting for corner markers...")
            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        warped, M = warp_perspective_with_corners(frame, marker_map)

        # --- TSP Points ---
# #         current_ids = set(marker_map.keys()) - set(CORNER_IDS) - {MOVING_ID}
# #         tsp_points = get_tsp_points(marker_map)
# #         if current_ids != last_marker_ids and len(tsp_points) >= 2:
# #             path, _ = compute_tsp_with_convex_hull(tsp_points)
# #             warped = draw_path_on_image(warped, tsp_points, path)
# #             last_marker_ids = current_ids

        # --- Moving Marker Tracking (ID 10) ---
        if MOVING_ID in marker_map:
            pts = marker_map[MOVING_ID][0]
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            pt = np.array([[cx, cy]], dtype="float32")
            warped_pt = cv2.perspectiveTransform(np.array([pt]), M)[0][0]
            trail.append(tuple(warped_pt.astype(int)))

        draw_trail(warped, trail)

        # --- Overlay Info ---
        fps = 1.0 / (time() - prev_time)
        prev_time = time()
        overlay_text(warped, f"FPS: {fps:.2f}")
        overlay_text(warped, f"Trail length: {len(trail)}", position=(10, 40))

        # --- Show ---
        cv2.imshow("Warped Area View", warped)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    picam2.stop()


if __name__ == "__main__":
    main()