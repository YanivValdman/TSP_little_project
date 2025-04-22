import cv2
import numpy as np
from time import time, sleep
from collections import deque
from picamera2 import Picamera2, Preview
from utils import detect_aruco_markers

# === CONFIG ===
CORNER_IDS = [1, 2, 3, 4]  # Required for perspective warping
MOVING_ID = 10             # The marker being tracked
MAX_TRAIL = 100            # Maximum trail length to display


def warp_perspective_with_corners(image, marker_map):
    """Transform the image based on the corner markers to get a top-down view"""
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


def draw_trail(image, trail):
    """Draw the movement trail of the tracked marker"""
    for i in range(1, len(trail)):
        cv2.line(image, trail[i-1], trail[i], (0, 0, 255), 2)
    if trail:
        cv2.circle(image, trail[-1], 5, (0, 0, 255), -1)


def main():
    # Initialize camera
    picam2 = Picamera2()
    picam2.start_preview(Preview.QTGL)
    config = picam2.create_preview_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    sleep(1)

    # Initialize tracking trail
    trail = deque(maxlen=MAX_TRAIL)

    count = 0
    while True:
        print(f"{count=}")
        count = count + 1
        # Capture frame and detect markers
        frame = picam2.capture_array("main")
        marker_map, _ = detect_aruco_markers(frame)

        # Check if all corner markers are visible
        missing_markers = [cid for cid in CORNER_IDS if cid not in marker_map]
        if missing_markers:
            cv2.putText(frame, f"Missing markers: {missing_markers}", (10, 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show which markers are detected
            for id, corners in marker_map.items():
                corners = corners[0].astype(int)
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                cv2.putText(frame, str(id), tuple(corners[0]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add instruction text
            cv2.putText(frame, "Press any key to refresh, 'q' to quit", (10, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
            cv2.imshow("Live Feed", frame)
            
            print("retrying")
            # Wait for key press, but with a value to capture
            # Brief non-blocking key check (1ms)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        else:
            # All corners detected, proceed with tracking
            warped, M = warp_perspective_with_corners(frame, marker_map)
            
            # Track the moving marker
            if MOVING_ID in marker_map:
                pts = marker_map[MOVING_ID][0]
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                pt = np.array([[cx, cy]], dtype="float32")
                warped_pt = cv2.perspectiveTransform(np.array([pt]), M)[0][0]
                trail.append(tuple(warped_pt.astype(int)))

            # Draw the tracking trail
            draw_trail(warped, trail)
            
            cv2.imshow("Tracking View", warped)
            
            # Check for quit key
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()