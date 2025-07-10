import cv2
import numpy as np
import pygame
from time import sleep
import threading
from collections import deque
from picamera2 import Picamera2
from utils import detect_aruco_markers
from heuristic_tsp import detect_tsp_points_in_warped_image

# === CONFIG ===
MOVING_ID = 10             # The marker being tracked
MAX_TRAIL = 1000           # Maximum trail length to display
VISIT_RADIUS = 50          # Radius in pixels to consider a point "visited"

class ArucoTracker:
    def __init__(self):
        self.running = True
        self.trail = deque(maxlen=MAX_TRAIL)
        self.lock = threading.Lock()
        self.current_frame = None
        self.frame_ready = False

        # For path tracking
        self.all_points = None          # List of all (x, y) tuples (TSP points)
        self.visited_points = []        # Ordered list of visited (x, y) tuples
        self.path_complete = False
        self.returning_to_start = False

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("ArUco Tracker")

    def initialize_points(self, frame):
        """Detect TSP points on the first frame (warped or not, as required)."""
        self.all_points = detect_tsp_points_in_warped_image(frame)
        # Defensive: convert all to tuples of ints
        self.all_points = [tuple(map(int, pt)) for pt in self.all_points]

    def update_path_logic(self, id10_pos):
        """Update visited path logic with the current id10 position."""
        if self.path_complete or self.all_points is None or len(self.all_points) == 0:
            return

        id10_pos = tuple(map(int, id10_pos))

        # If not all points visited, check for new visits
        if not self.returning_to_start:
            for pt in self.all_points:
                if pt not in self.visited_points:
                    if np.linalg.norm(np.array(id10_pos) - np.array(pt)) < VISIT_RADIUS:
                        self.visited_points.append(pt)
            if len(self.visited_points) == len(self.all_points):
                self.returning_to_start = True

        # If all points visited, check if we've returned to the first point
        if self.returning_to_start and len(self.visited_points) > 0:
            first_pt = np.array(self.visited_points[0])
            if np.linalg.norm(np.array(id10_pos) - first_pt) < VISIT_RADIUS:
                self.path_complete = True

    def draw_path(self, image, id10_pos):
        """Draw the TSP visited path including the current segment."""
        if not self.visited_points:
            return

        # Draw lines between visited points
        for i in range(1, len(self.visited_points)):
            cv2.line(image,
                     self.visited_points[i-1],
                     self.visited_points[i],
                     (0, 255, 0), 2)

        # Draw line from last visited to current id10 position (unless path is complete)
        if not self.path_complete:
            cv2.line(image,
                     self.visited_points[-1],
                     id10_pos,
                     (0, 0, 255), 2)
            # If returning to start, show the final return segment
            if self.returning_to_start:
                cv2.line(image,
                         id10_pos,
                         self.visited_points[0],
                         (255, 0, 0), 2)

        # Draw the points (visited)
        for x, y in self.visited_points:
            cv2.circle(image, (x, y), 6, (0, 0, 255), -1)

        # Draw all points (unvisited in gray)
        for pt in self.all_points:
            if pt not in self.visited_points:
                cv2.circle(image, pt, 6, (180, 180, 180), 2)

        # Draw the current id10 position
        cv2.circle(image, id10_pos, 8, (0, 255, 255), -1)

    def capture_loop(self):
        """Thread function to capture and process frames"""
        # Initialize camera
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (800, 600)})
        picam2.configure(config)
        picam2.start()
        sleep(1)

        count = 0
        initialized_points = False

        try:
            while self.running:
                count += 1

                # Capture and process frame
                frame = picam2.capture_array("main")
                marker_map, _ = detect_aruco_markers(frame)

                # On first frame, detect TSP points
                if not initialized_points:
                    self.initialize_points(frame)
                    initialized_points = True

                # Track the moving marker
                id10_pos = None
                if MOVING_ID in marker_map:
                    pts = marker_map[MOVING_ID][0]
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    id10_pos = (cx, cy)
                    self.trail.append(id10_pos)
                else:
                    # No valid detection, skip update
                    display_frame = frame.copy()
                    with self.lock:
                        self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        self.frame_ready = True
                    sleep(0.1)
                    continue

                # Update the path/path logic
                self.update_path_logic(id10_pos)

                # Create a display frame
                display_frame = frame.copy()
                self.draw_path(display_frame, id10_pos)

                # Optionally: Draw the trail (remove if not needed)
                # self.draw_trail(display_frame, self.trail)

                # Update the current frame with lock to avoid race conditions
                with self.lock:
                    self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    self.frame_ready = True

                # Stop updating if path is complete
                if self.path_complete:
                    self.running = False

                sleep(0.1)
        finally:
            picam2.stop()
            print("Camera resources released")

    def draw_trail(self, image, trail):
        """Draw the movement trail of the tracked marker (optional)"""
        for i in range(1, len(trail)):
            cv2.line(image, trail[i-1], trail[i], (0, 0, 255), 2)
        if trail:
            cv2.circle(image, trail[-1], 5, (0, 0, 255), -1)

    def display_loop(self):
        """Thread function to handle displaying with Pygame"""
        clock = pygame.time.Clock()

        while self.running:
            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Update display if new frame is ready
            if self.frame_ready:
                with self.lock:
                    if self.current_frame is not None:
                        frame_to_display = self.current_frame.copy()
                        self.frame_ready = False
                    else:
                        continue

                # Convert OpenCV image to Pygame surface
                h, w = frame_to_display.shape[:2]
                surface = pygame.surfarray.make_surface(frame_to_display.swapaxes(0, 1))
                self.screen.blit(surface, (0, 0))
                pygame.display.flip()

            # Limit to 30 FPS
            clock.tick(30)

    def run(self):
        """Start the tracker with separate threads"""
        print("Starting ArUco tracker with Pygame display...")

        # Create and start threads
        capture_thread = threading.Thread(target=self.capture_loop)
        capture_thread.daemon = True

        display_thread = threading.Thread(target=self.display_loop)
        display_thread.daemon = True

        capture_thread.start()
        display_thread.start()

        # Wait for threads to complete
        try:
            while self.running:
                sleep(0.1)
        except KeyboardInterrupt:
            self.running = False

        print("Waiting for threads to finish...")
        capture_thread.join(timeout=2)
        display_thread.join(timeout=2)

        # Clean up resources
        pygame.quit()
        print("Tracker shutdown complete")

if __name__ == "__main__":
    tracker = ArucoTracker()
    tracker.run()