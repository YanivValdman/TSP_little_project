import cv2
import numpy as np
import pygame
from time import sleep
import threading
from collections import deque
from picamera2 import Picamera2
from utils import detect_aruco_markers
from heuristic_tsp import detect_tsp_points_in_warped_image, compute_tsp_with_convex_hull

# === CONFIG ===
MOVING_ID = 10
MAX_TRAIL = 1000
VISIT_RADIUS = 50

# UI constants
VIDEO_W, VIDEO_H = 800, 600
WIN_W, WIN_H = VIDEO_W * 2, VIDEO_H
RIGHT_W = VIDEO_W
INFO_H = int(VIDEO_H * 2 / 3)
BUTTON_H = VIDEO_H - INFO_H
BUTTON_PAD = 30
BUTTON_COLOR = (180, 180, 180)
BUTTON_COLOR_ACTIVE = (90, 180, 90)
BUTTON_FONT_SIZE = 36
INFO_FONT_SIZE = 32

# COLOR PREFERENCES
PATH_COLOR = (0, 0, 255)             # Blue
CLOSING_LINE_COLOR = (100, 200, 255) # Light Blue
VISITED_PT_COLOR = (255, 0, 0)       # Red
SOLUTION_COLOR = (0, 255, 0)         # Green
SOLUTION_PT_COLOR = (255, 0, 0)      # Red
ID10_COLOR = (255, 215, 0)           # Gold (for visibility)

class ArucoTracker:
    def __init__(self):
        self.running = True
        self.trail = deque(maxlen=MAX_TRAIL)
        self.lock = threading.Lock()
        self.current_frame = None
        self.frame_ready = False

        # For path tracking
        self.all_points = None
        self.visited_points = []
        self.path_complete = False
        self.returning_to_start = False
        self.show_solution = False
        self.optimal_path = None
        self.optimal_path_ready = False
        self.pixels_per_meter = None

        # For UI state
        self.video_surf = None

        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("ArUco Tracker with UI")
        self.button_font = pygame.font.SysFont(None, BUTTON_FONT_SIZE)
        self.info_font = pygame.font.SysFont(None, INFO_FONT_SIZE)
        self.clock = pygame.time.Clock()

        # Define buttons (rects for mouse events)
        self.button_restart = pygame.Rect(VIDEO_W + BUTTON_PAD, INFO_H + BUTTON_PAD, (RIGHT_W - 3 * BUTTON_PAD) // 2, BUTTON_H - 2 * BUTTON_PAD)
        self.button_solution = pygame.Rect(self.button_restart.right + BUTTON_PAD, INFO_H + BUTTON_PAD, (RIGHT_W - 3 * BUTTON_PAD) // 2, BUTTON_H - 2 * BUTTON_PAD)

    def initialize_points_and_calibration(self, frame):
        self.all_points = detect_tsp_points_in_warped_image(frame, MOVING_ID)
        self.all_points = [tuple(map(int, pt)) for pt in self.all_points]
        marker_map, _ = detect_aruco_markers(frame)
        try:
            p1 = marker_map[1][0]
            p2 = marker_map[2][0]
            cx1, cy1 = int(np.mean(p1[:, 0])), int(np.mean(p1[:, 1]))
            cx2, cy2 = int(np.mean(p2[:, 0])), int(np.mean(p2[:, 1]))
            pixel_dist = np.linalg.norm(np.array([cx1, cy1]) - np.array([cx2, cy2]))
            self.pixels_per_meter = pixel_dist / 1.3
        except Exception as e:
            self.pixels_per_meter = 1.0
            print("Calibration failed:", e)

    def update_path_logic(self, id10_pos):
        if self.path_complete or self.all_points is None or len(self.all_points) == 0:
            return

        id10_pos = tuple(map(int, id10_pos))

        # If not all points visited, check for new visits
        if not self.returning_to_start:
            for pt in self.all_points:
                if pt not in self.visited_points and np.linalg.norm(np.array(id10_pos) - np.array(pt)) < VISIT_RADIUS:
                    self.visited_points.append(pt)
            if len(self.visited_points) == len(self.all_points):
                self.returning_to_start = True

        # If all points visited, check if we've returned to the first point
        if self.returning_to_start and len(self.visited_points) > 0:
            first_pt = np.array(self.visited_points[0])
            if np.linalg.norm(np.array(id10_pos) - first_pt) < VISIT_RADIUS:
                self.path_complete = True
                self.show_solution = True

    def draw_path(self, surf, id10_pos):
        if not self.visited_points:
            return

        # Draw blue lines between visited points
        for i in range(1, len(self.visited_points)):
            pygame.draw.line(surf, PATH_COLOR, self.visited_points[i-1], self.visited_points[i], 4)

        # Draw visited points as red circles
        for x, y in self.visited_points:
            pygame.draw.circle(surf, VISITED_PT_COLOR, (x, y), 8)

        if not self.path_complete:
            if len(self.visited_points) > 0 and id10_pos is not None:
                # Draw blue line from last visited point to id10 (current position)
                pygame.draw.line(surf, PATH_COLOR, self.visited_points[-1], id10_pos, 4)
            if self.returning_to_start and not self.path_complete:
                # Draw light blue line from id10 to first point
                pygame.draw.line(surf, CLOSING_LINE_COLOR, id10_pos, self.visited_points[0], 4)
            # Draw id10 marker
            pygame.draw.circle(surf, ID10_COLOR, id10_pos, 12)
        else:
            # When path is complete, draw closing segment (last point to first)
            if len(self.visited_points) > 1:
                pygame.draw.line(surf, PATH_COLOR, self.visited_points[-1], self.visited_points[0], 4)
            # Do NOT draw id10 marker

    def draw_solution(self, surf):
        if not self.show_solution or not self.optimal_path_ready or not self.optimal_path:
            return
        # Draw green lines for solution path
        for i in range(1, len(self.optimal_path)):
            pygame.draw.line(surf, SOLUTION_COLOR,
                             tuple(map(int, self.optimal_path[i-1])),
                             tuple(map(int, self.optimal_path[i])), 4)
        pygame.draw.line(surf, SOLUTION_COLOR,
                         tuple(map(int, self.optimal_path[-1])),
                         tuple(map(int, self.optimal_path[0])), 4)
        # Draw red points for TSP points
        for x, y in self.optimal_path:
            pygame.draw.circle(surf, SOLUTION_PT_COLOR, (int(x), int(y)), 8)

    def compute_total_distance(self):
        if self.pixels_per_meter is None or len(self.visited_points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.visited_points)):
            total += np.linalg.norm(np.array(self.visited_points[i]) - np.array(self.visited_points[i-1]))
        return total / self.pixels_per_meter

    def compute_optimal_path_distance(self):
        """Compute the total length in meters of the optimal TSP path."""
        if not self.optimal_path or len(self.optimal_path) < 2 or not self.pixels_per_meter:
            return 0.0
        total = 0.0
        for i in range(1, len(self.optimal_path)):
            total += np.linalg.norm(np.array(self.optimal_path[i]) - np.array(self.optimal_path[i-1]))
        # Closing the loop:
        total += np.linalg.norm(np.array(self.optimal_path[0]) - np.array(self.optimal_path[-1]))
        return total / self.pixels_per_meter

    def capture_loop(self):
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (VIDEO_W, VIDEO_H)})
        picam2.configure(config)
        picam2.start()
        sleep(1)
        initialized_points = False

        try:
            while self.running:
                frame = picam2.capture_array("main")
                marker_map, _ = detect_aruco_markers(frame)

                if not initialized_points:
                    self.initialize_points_and_calibration(frame)
                    if self.all_points:
                        opt_path, _ = compute_tsp_with_convex_hull(self.all_points)
                        self.optimal_path = [tuple(map(int, pt)) for pt in opt_path]
                        self.optimal_path_ready = True
                    initialized_points = True

                id10_pos = None
                if MOVING_ID in marker_map:
                    pts = marker_map[MOVING_ID][0]
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    id10_pos = (cx, cy)
                    if not self.path_complete:
                        self.trail.append(id10_pos)
                else:
                    with self.lock:
                        self.current_frame = frame.copy()
                        self.frame_ready = True
                    sleep(0.1)
                    continue

                if not self.path_complete and id10_pos is not None:
                    self.update_path_logic(id10_pos)

                with self.lock:
                    self.current_frame = frame.copy()
                    self.frame_ready = True

                sleep(0.1)
        finally:
            picam2.stop()
            print("Camera resources released")

    def draw_buttons(self):
        color = BUTTON_COLOR_ACTIVE if self.restart_hovered else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, self.button_restart)
        restart_label = self.button_font.render("Restart", True, (0, 0, 0))
        self.screen.blit(restart_label, (
            self.button_restart.centerx - restart_label.get_width() // 2,
            self.button_restart.centery - restart_label.get_height() // 2
        ))

        color = BUTTON_COLOR_ACTIVE if self.solution_hovered or self.show_solution else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, self.button_solution)
        solution_label = self.button_font.render("Show Solution", True, (0, 0, 0))
        self.screen.blit(solution_label, (
            self.button_solution.centerx - solution_label.get_width() // 2,
            self.button_solution.centery - solution_label.get_height() // 2
        ))

    def draw_info(self):
        y = BUTTON_PAD
        linespacing = INFO_FONT_SIZE + 10
        dist_m = self.compute_total_distance()
        dist_str = f"Total Distance: {dist_m:.2f} m"
        dist_label = self.info_font.render(dist_str, True, (50, 50, 50))
        self.screen.blit(dist_label, (VIDEO_W + BUTTON_PAD, y))
        y += linespacing

        num_visited = len(self.visited_points)
        num_total = len(self.all_points) if self.all_points else 0
        pts_str = f"Points Visited: {num_visited} / {num_total}"
        pts_label = self.info_font.render(pts_str, True, (50, 50, 50))
        self.screen.blit(pts_label, (VIDEO_W + BUTTON_PAD, y))
        y += linespacing

        # Only show optimal TSP path distance if solution is toggled on, and show it below 'Points Visited'
        if self.show_solution:
            opt_dist_m = self.compute_optimal_path_distance()
            opt_dist_str = f"Optimal Path: {opt_dist_m:.2f} m"
            opt_dist_label = self.info_font.render(opt_dist_str, True, (20, 100, 20))
            self.screen.blit(opt_dist_label, (VIDEO_W + BUTTON_PAD, y))
            y += linespacing

    def display_loop(self):
        self.restart_hovered = False
        self.solution_hovered = False

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.button_restart.collidepoint(event.pos):
                        self.restart()
                    elif self.button_solution.collidepoint(event.pos):
                        self.show_solution = not self.show_solution
                if event.type == pygame.MOUSEMOTION:
                    self.restart_hovered = self.button_restart.collidepoint(event.pos)
                    self.solution_hovered = self.button_solution.collidepoint(event.pos)

            if self.frame_ready:
                with self.lock:
                    frame = None
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                if frame is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    surf = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
                    surf = pygame.transform.scale(surf, (VIDEO_W, VIDEO_H))
                    self.video_surf = surf

            self.screen.fill((220, 220, 220))

            if self.video_surf:
                self.screen.blit(self.video_surf, (0, 0))

                overlay = pygame.Surface((VIDEO_W, VIDEO_H), pygame.SRCALPHA)
                if self.visited_points and self.trail:
                    self.draw_path(
                        overlay,
                        self.trail[-1] if self.trail else self.visited_points[-1]
                    )
                self.draw_solution(overlay)
                self.screen.blit(overlay, (0, 0))

            self.draw_info()
            self.draw_buttons()

            pygame.display.flip()
            self.clock.tick(30)
            if not self.running:
                break

        pygame.quit()

    def restart(self):
        self.trail.clear()
        self.visited_points = []
        self.path_complete = False
        self.returning_to_start = False
        self.show_solution = False

    def run(self):
        print("Starting ArUco tracker with Pygame UI...")

        capture_thread = threading.Thread(target=self.capture_loop)
        capture_thread.daemon = True
        capture_thread.start()

        self.display_loop()

        print("Waiting for threads to finish...")
        capture_thread.join(timeout=2)
        print("Tracker shutdown complete")

if __name__ == "__main__":
    tracker = ArucoTracker()
    tracker.run()