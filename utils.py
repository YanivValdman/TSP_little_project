from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import cv2


def compute_tsp_with_convex_hull(points):
    # Step 1: Compute the Convex Hull
    hull = ConvexHull(points)
    hull_path = list(hull.vertices)

    # Step 2: Order the Convex Hull points into a cycle
    hull_path.append(hull.vertices[0])

    # Step 3: Iterate over remaining points and insert them into the hull
    remaining_points = set(range(len(points))) - set(hull.vertices)
    path = [points[v] for v in hull_path]

    while remaining_points:
        best_ratio = float("inf")
        best_point_idx = None
        insert_idx = 0

        for free_idx in remaining_points:
            free_point = points[free_idx]

            # Find the best insertion point in the current path
            for i in range(len(path) - 1):
                current_cost = euclidean(path[i], path[i + 1])
                new_cost = (
                    euclidean(path[i], free_point)
                    + euclidean(free_point, path[i + 1])
                )
                cost_ratio = new_cost / current_cost

                if cost_ratio < best_ratio:
                    best_ratio = cost_ratio
                    best_point_idx = free_idx
                    insert_idx = i + 1

        # Insert the best point into the path
        path.insert(insert_idx, points[best_point_idx])
        remaining_points.remove(best_point_idx)

    # Step 4: Compute the total cost of the final path
    total_cost = sum(euclidean(path[i], path[i + 1]) for i in range(len(path) - 1))

    return path, total_cost


def plot_tsp_path(points, path):
    # Ensure points and path are NumPy arrays for comparison
    points = np.array(points)
    path = np.array(path)

    # Convert path to indices for easier plotting
    try:
        path_indices = [np.where((points == p).all(axis=1))[0][0] for p in path]
    except AttributeError:
        raise ValueError("Ensure `points` and `path` have matching structures and types.")

    # Prepare the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], color="red", label="Points")
    plt.plot(
        [p[0] for p in path],
        [p[1] for p in path],
        color="blue",
        label="TSP Path",
        marker="o",
    )

    for i, point in enumerate(points):
        plt.text(point[0], point[1], f"{i}", fontsize=8, ha="right")

    plt.title("TSP Solution Using Convex Hull")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()
    
def draw_path_on_image(image, points, path):
    """
    Draw the TSP solution path on a copy of the image.

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

    # Draw the points with labels
    for idx, (x, y) in enumerate(points):
        cv2.circle(image_with_path, (int(x), int(y)), 6, (0, 0, 255), -1)  # Red circle
        cv2.putText(
            image_with_path,
            str(idx),
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )  # White text

    return image_with_path


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
    