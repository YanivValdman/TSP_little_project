import cv2
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import euclidean
from itertools import permutations
from utils import draw_path_on_image, capture_image, detect_aruco_markers, detect_corners_and_warp


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



def detect_tsp_points_in_warped_image(warped_image, car_id):
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
        if id in corner_ids or id == car_id:
            continue  # Skip corner markers

        # Compute the center of the marker
        pts = corners[0]
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        tsp_points.append((cx, cy))

    return tsp_points


# --- Main Execution ---
def main():
    car_id = 10
    expected_points = 3
    image = capture_image()
    cv2.imwrite("raw.jpg", image)
    if image is None:
        print("Error loading image.")
        return

    try:
        # Step 1: Detect corners and warp the image
        
        #warped, M = detect_corners_and_warp(image)
        warped = cv2.resize(image, (0, 0), fx = 0.7, fy = 0.7) #added by TOM

        # Step 2: Detect TSP points in the warped image
        points = detect_tsp_points_in_warped_image(warped, car_id) 
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
    image_with_path = cv2.resize(image_with_path, (0, 0), fx = 0.5, fy = 0.5) #added by TOM

    # Display or save the result
    cv2.imshow("TSP Path on Warped Image", image_with_path)
    cv2.imwrite("tsp_solution.png", image_with_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()