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


def get_perspective_transform(image, marker_map):
    corner_ids = [1, 2, 3, 4]
    if not all(cid in marker_map for cid in corner_ids):
        raise Exception("Missing one or more corner markers (IDs 1–4)")

    centers = []
    for cid in corner_ids:
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


def get_tsp_points(marker_map, M):
    """
    Extract TSP points that fall within the rectangle defined by corner markers.

    Args:
        marker_map (dict): Dictionary mapping marker IDs to their corner points.
        M (np.ndarray): Perspective transformation matrix.

    Returns:
        list: List of valid TSP points as (x, y) tuples.
    """
    # Get the corner IDs
    corner_ids = [1, 2, 3, 4]
    rectangle_points = []

    # Ensure all corner IDs are present
    for cid in corner_ids:
        if cid in marker_map:
            pts = marker_map[cid][0]
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            rectangle_points.append((cx, cy))
        else:
            raise Exception(f"Missing corner marker with ID {cid}")

    print(f"{rectangle_points = }")
    
    
    # Compute the bounding box of the rectangle
    min_x = min(p[0] for p in rectangle_points)
    max_x = max(p[0] for p in rectangle_points)
    min_y = min(p[1] for p in rectangle_points)
    max_y = max(p[1] for p in rectangle_points)
    
    print(f"{min_x = }")
    print(f"{max_x = }")
    print(f"{min_y = }")
    print(f"{max_y = }")

    tsp_points = []
    for id, corners in marker_map.items():
        if id in corner_ids:
            continue  # Skip corner markers

        # Calculate the center of the marker
        pts = corners[0]
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        pt = np.array([[[cx, cy]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pt, M)[0][0]
        wx, wy = int(warped_pt[0]), int(warped_pt[1])
        print(f"{id = }")
        print(f"{wx = }")
        print(f"{wy = }")

        # Check if the point lies within the bounding box
        if min_x <= wx <= max_x and min_y <= wy <= max_y:
            tsp_points.append((wx, wy))  # Add only valid points

    return tsp_points


def tsp_bruteforce(points):
    distances = {}
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                distances[(i, j)] = euclidean(p1, p2)

    min_path = None
    min_cost = float("inf")
    for perm in permutations(range(len(points))):
        cost = sum(distances[(perm[i], perm[i + 1])] for i in range(len(perm) - 1))
        if cost < min_cost:
            min_cost = cost
            min_path = perm
    return min_path, min_cost


def draw_path(image, points, path):
    for i in range(len(path) - 1):
        cv2.line(image, points[path[i]], points[path[i + 1]], (129, 193, 161), 2)
    for idx, (x, y) in enumerate(points):
        cv2.circle(image, (x, y), 6, (45, 127, 254), -1)
        cv2.putText(image, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 50), 1)
    return image

# --- Main Execution ---
def main():
    image = cv2.imread("raw.jpg")  # replace with camera capture if needed
    expected_points = 6
#     image = capture_image()
#     cv2.imwrite("raw.jpg", image)
    if image is None:
        print("Error loading image.")
        return

    marker_map, corners = detect_aruco_markers(image)

    try:
        warped, M = get_perspective_transform(image, marker_map)
    except Exception as e:
        print("Perspective transform error:", e)
        return

    points = get_tsp_points(marker_map, M)
    print("points")
    print(points)
    
    if len(points) != expected_points:
        print("Not the expected number of points. Expected:", expected_points, "But got:", len(points))
        return
    
    if len(points) < 2:
        print("Not enough TSP points.")
        return

#     path, cost = tsp_bruteforce(points)
    path, cost = compute_tsp_with_convex_hull(points)
    print("Optimal Path:", path)
    print("Total Distance:", cost)

     # Draw the TSP path on the image
    image_with_path = draw_path_on_image(warped, points, path)

    # Display or save the result
    cv2.imshow("TSP Path on Image", image_with_path)
    cv2.imwrite("tsp_solution.png", image_with_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_single_aruco_marker( marker_dict=cv2.aruco.DICT_5X5_50):
    captured = capture_image()
    image = cv2.cvtColor(captured, cv2.COLOR_RGB2BGR)
    cv2.imwrite("raw.jpg", image)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the ArUco dictionary and parameters
    aruco_dict = cv2.aruco.Dictionary_get(marker_dict)
    parameters = cv2.aruco.DetectorParameters_create()

    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None or len(ids) == 0:
        print("No markers detected.")
        return None

    if len(ids) > 1:
        print("Multiple markers detected. Only processing the first one.")

    # Draw the detected marker
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    # Optional: Show the image
    # cv2.imshow("Detected ArUco Marker", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Return ID and corner positions of the first marker
    return {
        "id": int(ids[0][0]),
        "corners": corners[0].tolist()
    }
    
if __name__ == "__main__":
    main()
    #print(detect_single_aruco_marker())
    #print("done")
