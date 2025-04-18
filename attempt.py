import cv2
import numpy as np
from time import sleep
from scipy.spatial.distance import euclidean
from itertools import permutations
from picamera2 import MappedArray, Picamera2, Preview
from utils import compute_tsp_with_convex_hull, plot_tsp_path, draw_path_on_image

def capture_image():
    picam2 = Picamera2()
    picam2.start_preview(Preview.QTGL)
    config = picam2.create_preview_configuration(main={"size": (4056, 3040)})
    picam2.configure(config)
    picam2.start()
    sleep(1)
    captured = picam2.capture_array("main")
    picam2.stop()
    return captured


def detect_aruco_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None:
        return {}, []
    ids = ids.flatten()
    marker_map = dict(zip(ids, corners))
    return marker_map, corners

def detect_corners_and_warp(image):
    """
    Detect the four corner markers (IDs 1-4) and warp the image to a square.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The warped image.
        np.ndarray: The perspective transformation matrix.
    """
    marker_map, _ = detect_aruco_markers(image)

    # Detect only the corner markers
    corner_ids = [1, 2, 3, 4]
    if not all(cid in marker_map for cid in corner_ids):
        raise Exception("Missing one or more corner markers (IDs 1–4)")

    # Compute the centers of the corner markers
    centers = []
    for cid in corner_ids:
        pts = marker_map[cid][0]
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        centers.append((cx, cy))

    # Sort and order the centers to form the rectangle
    centers = sorted(centers, key=lambda p: (p[1], p[0]))
    top = sorted(centers[:2], key=lambda p: p[0])
    bottom = sorted(centers[2:], key=lambda p: p[0])
    ordered = np.array([top[0], top[1], bottom[0], bottom[1]], dtype="float32")

    # Define the destination points for the warped image
    dst = np.array([[0, 0], [500, 0], [0, 500], [500, 500]], dtype="float32")

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(ordered, dst)

    # Warp the image
    warped = cv2.warpPerspective(image, M, (500, 500))

    return warped, M


def detect_tsp_points_in_warped_image(warped_image):
    """
    Detect TSP points (non-corner markers) in the warped image.

    Args:
        warped_image (np.ndarray): The warped image.

    Returns:
        list: List of TSP points as (x, y) tuples.
    """
    marker_map, _ = detect_aruco_markers(warped_image)

    # Exclude corner markers (IDs 1-4)
    corner_ids = [1, 2, 3, 4]
    tsp_points = []
    for id, corners in marker_map.items():
        if id in corner_ids:
            continue  # Skip corner markers

        # Compute the center of the marker
        pts = corners[0]
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        tsp_points.append((cx, cy))

    return tsp_points


# --- Main Execution ---
def main():
    image = cv2.imread("raw.jpg")  # Replace with camera capture if needed
    expected_points = 6
    if image is None:
        print("Error loading image.")
        return

    try:
        # Step 1: Detect corners and warp the image
        warped, M = detect_corners_and_warp(image)

        # Step 2: Detect TSP points in the warped image
        points = detect_tsp_points_in_warped_image(warped)
    except Exception as e:
        print("Error:", e)
        return

    # Step 3: Validate detected points
    if len(points) != expected_points:
        print(f"Not the expected number of points. Expected: {expected_points}, Got: {len(points)}")
        return

    if len(points) < 2:
        print("Not enough TSP points.")
        return

    # Step 4: Solve TSP and display the result
    path, cost = compute_tsp_with_convex_hull(points)
    print("Optimal Path:", path)
    print("Total Distance:", cost)

    # Step 5: Draw the TSP path on the warped image
    image_with_path = draw_path_on_image(warped, points, path)

    # Display or save the result
    cv2.imshow("TSP Path on Warped Image", image_with_path)
    cv2.imwrite("tsp_solution.png", image_with_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()