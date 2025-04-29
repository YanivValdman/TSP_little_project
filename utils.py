import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import cv2
from picamera2 import MappedArray, Picamera2, Preview
from time import sleep


# ========================= Image & AruCo Related Utils ================================

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


# ======================================================================================



# def plot_tsp_path(points, path):
#     # Ensure points and path are NumPy arrays for comparison
#     points = np.array(points)
#     path = np.array(path)
# 
#     # Convert path to indices for easier plotting
#     try:
#         path_indices = [np.where((points == p).all(axis=1))[0][0] for p in path]
#     except AttributeError:
#         raise ValueError("Ensure `points` and `path` have matching structures and types.")
# 
#     # Prepare the plot
#     plt.figure(figsize=(8, 6))
#     plt.scatter(points[:, 0], points[:, 1], color="red", label="Points")
#     plt.plot(
#         [p[0] for p in path],
#         [p[1] for p in path],
#         color="blue",
#         label="TSP Path",
#         marker="o",
#     )
# 
#     for i, point in enumerate(points):
#         plt.text(point[0], point[1], f"{i}", fontsize=8, ha="right")
# 
#     plt.title("TSP Solution Using Convex Hull")
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.legend()
#     plt.grid()
#     plt.show()
    

def draw_path_on_image(image, points, path):
    """
    Draw the TSP solution path on a copy of the image without point labels.

    Args:
        image (np.ndarray): The original image.
        points (list): List of TSP points as (x, y) tuples.
        path (list): The computed TSP path as a list of (x, y) tuples.

    Returns:
        np.ndarray: The image with the path drawn on it.
    """
    # Make a copy of the image to draw on
    image_with_path = image.copy()

    # Draw the TSP path
    for i in range(len(path) - 1):
        start_point = tuple(map(int, path[i]))
        end_point = tuple(map(int, path[i + 1]))
        cv2.line(image_with_path, start_point, end_point, (0, 255, 0), 2)  # Green line

    # Draw the points without labels
    for x, y in points:
        cv2.circle(image_with_path, (int(x), int(y)), 6, (0, 0, 255), -1)  # Red circle
        # Text labels removed

    return image_with_path

if __name__ == "__main__":
    points = np.array([[0.80007683, 0.20169368],
 [0.43482329, 0.03859274],
 [0.31029877, 0.43831088],
 [0.8216282,  0.90369553],
 [0.42425499, 0.21624647],
 [0.07117709, 0.16876709],
 [0.05239259, 0.25369135],
 [0.11108453, 0.81352358]])
    path, total_cost = compute_tsp_with_convex_hull(points)
    plot_tsp_path(points, path)
    print("Optimal Path:", path)
    print("Total Cost:", total_cost)
    