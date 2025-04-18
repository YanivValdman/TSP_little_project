from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


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
    # Convert path to indices for easier plotting
    path_indices = [np.where((points == p).all(axis=1))[0][0] for p in path]

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
    