import cv2
import numpy as np
import pygame
from time import sleep
import threading
from collections import deque
import queue
from picamera2 import Picamera2
from utils import detect_aruco_markers
from heuristic_tsp import detect_tsp_points_in_warped_image, compute_tsp_with_convex_hull
import time
import config
import sys

# === CONFIG ===
MAX_TRAIL = 1000
VISIT_RADIUS = 50
TSP_POINT_ERROR_RETRY_SECONDS = 5
CORNERS_ERROR_RETRY_SECONDS = 5

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
MENU_FONT_SIZE = 50

PATH_COLOR = (0, 0, 255)
CLOSING_LINE_COLOR = (100, 200, 255)
VISITED_PT_COLOR = (255, 0, 0)
SOLUTION_COLOR = (0, 255, 0)
SOLUTION_PT_COLOR = (255, 0, 0)
ID10_COLOR = (255, 215, 0)

# Store the current expected points in a global variable for runtime use
CURRENT_EXPECTED_POINTS = config.EXPECTED_TSP_POINTS

def show_wrapped_message(screen, message, font, color, bg, max_width, duration_ms=1000):
    """Display a wrapped message in the center of the window for a set duration."""
    screen.fill(bg)
    lines = []
    words = message.split(' ')
    current_line = ""
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        test_surface = font.render(test_line, True, color)
        if test_surface.get_width() <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    total_height = sum(font.render(line, True, color).get_height() + 2 for line in lines)
    y = (screen.get_height() - total_height) // 2
    for line in lines:
        line_surface = font.render(line, True, color)
        x = (screen.get_width() - line_surface.get_width()) // 2
        screen.blit(line_surface, (x, y))
        y += line_surface.get_height() + 2

    pygame.display.flip()
    pygame.time.wait(duration_ms)

def show_menu_and_get_expected_points():
    global CURRENT_EXPECTED_POINTS
    pygame.init()
    screen = pygame.display.set_mode((600, 300))
    pygame.display.set_caption("Startup Menu")
    font = pygame.font.SysFont(None, MENU_FONT_SIZE)
    small_font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()

    options = ["Use default configuration", "Use custom number of points"]
    selected = 0
    menu_active = True
    custom_points = None
    entering_number = False
    input_number_str = ""
    menu_start_time = time.time()
    timeout = 10  # seconds

    def draw_wrapped_text(text, font, color, x, y, max_width, surface):
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + (' ' if current_line else '') + word
            test_surface = font.render(test_line, True, color)
            if test_surface.get_width() <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        for line in lines:
            line_surface = font.render(line, True, color)
            surface.blit(line_surface, (x, y))
            y += line_surface.get_height() + 2
        return y

    while menu_active:
        screen.fill((240, 240, 240))
        elapsed = time.time() - menu_start_time
        timeout_rem = timeout - int(elapsed)
        if entering_number:
            prompt = "Enter number of points (2-30): " + input_number_str
            prompt_surface = font.render(prompt, True, (0, 0, 0))
            screen.blit(prompt_surface, (50, 100))
            info_surface = small_font.render("Press Enter to confirm. ESC to cancel.", True, (100, 100, 100))
            screen.blit(info_surface, (50, 160))
        else:
            for i, option in enumerate(options):
                color = (0, 0, 0)
                bg = BUTTON_COLOR_ACTIVE if i == selected else BUTTON_COLOR
                rect = pygame.Rect(50, 80 + i*70, 500, 60)
                pygame.draw.rect(screen, bg, rect)
                text = font.render(option, True, color)
                screen.blit(text, (rect.x + 15, rect.y + 10))
            # Timeout info (wrapped)
            info = f"Auto-selecting default in {timeout_rem}s..." if timeout_rem > 0 else "Auto-selected default."
            info_col = (100, 60, 60) if timeout_rem > 0 else (0, 120, 0)
            draw_wrapped_text(info, small_font, info_col, 50, 30, 500, screen)

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

            if entering_number:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        entering_number = False
                        input_number_str = ""
                    elif event.key == pygame.K_BACKSPACE:
                        input_number_str = input_number_str[:-1]
                    elif event.key == pygame.K_RETURN:
                        try:
                            val = int(input_number_str)
                            if 2 <= val <= 30:
                                custom_points = val
                                menu_active = False
                                break
                            else:
                                input_number_str = ""
                        except:
                            input_number_str = ""
                    elif event.unicode.isdigit():
                        if len(input_number_str) < 2:
                            input_number_str += event.unicode
            else:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(options)
                    elif event.key == pygame.K_UP:
                        selected = (selected - 1) % len(options)
                    elif event.key == pygame.K_RETURN:
                        if selected == 0:
                            menu_active = False
                            break
                        elif selected == 1:
                            entering_number = True
                            input_number_str = ""
        if not entering_number and elapsed >= timeout:
            # Timeout: select default automatically
            selected = 0
            menu_active = False
            # Show auto-selected message for 1 second before starting, WRAPPED
            show_wrapped_message(
                screen,
                "Default configuration selected automatically.",
                font, (0, 120, 0), (240, 240, 240), 500, duration_ms=1000
            )
            break

    # Set CURRENT_EXPECTED_POINTS
    if custom_points:
        CURRENT_EXPECTED_POINTS = custom_points
    else:
        CURRENT_EXPECTED_POINTS = config.EXPECTED_TSP_POINTS

    pygame.display.quit()
    pygame.quit()
    return CURRENT_EXPECTED_POINTS


class ArucoTracker:
    def __init__(self):
        self.running = True
        self.trail = deque(maxlen=MAX_TRAIL)
        
        # Replace shared variables with thread-safe queues
        self.frame_queue = queue.Queue(maxsize=2)  # Limit queue size to prevent memory buildup
        self.trail_update_queue = queue.Queue()    # For trail updates from capture thread
        self.restart_queue = queue.Queue()         # For restart commands
        
        # Thread-safe lock for trail access (still needed for trail manipulation)
        self.trail_lock = threading.Lock()
        
        self.picam2 = None  # Camera will be initialized in main thread


        # For path tracking

        self.all_points = None
        self.visited_points = []
        self.path_complete = False
        self.returning_to_start = False
        self.show_solution = False
        self.optimal_path = None
        self.optimal_path_ready = False
        self.pixels_per_meter = None


        self.tsp_error_msg = ""
        self.tsp_error_wait_until = 0
        self.tsp_error_active = False

        self.corners_error_msg = ""
        self.corners_error_wait_until = 0
        self.corners_error_active = False

         # For UI state
        self.video_surf = None
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("ArUco Tracker with UI")
        self.button_font = pygame.font.SysFont(None, BUTTON_FONT_SIZE)
        self.info_font = pygame.font.SysFont(None, INFO_FONT_SIZE)

        self.error_font = pygame.font.SysFont(None, 48)
        self.clock = pygame.time.Clock()
        # Define buttons (rects for mouse events)

        self.button_restart = pygame.Rect(VIDEO_W + BUTTON_PAD, INFO_H + BUTTON_PAD, (RIGHT_W - 3 * BUTTON_PAD) // 2, BUTTON_H - 2 * BUTTON_PAD)
        self.button_solution = pygame.Rect(self.button_restart.right + BUTTON_PAD, INFO_H + BUTTON_PAD, (RIGHT_W - 3 * BUTTON_PAD) // 2, BUTTON_H - 2 * BUTTON_PAD)

    def initialize_points_and_calibration(self, frame):
        marker_map, _ = detect_aruco_markers(frame)
        try:
            required_ids = [1, 2, 3, 4]
            if not all(id_ in marker_map for id_ in required_ids):
                self.corners_error_msg = "ERROR: Did not detect 4 corners."
                self.corners_error_active = True
                self.corners_error_wait_until = time.time() + CORNERS_ERROR_RETRY_SECONDS
                return False
            else:
                self.corners_error_msg = ""
                self.corners_error_active = False

            self.all_points = detect_tsp_points_in_warped_image(frame, config.MOVING_ID)
            self.all_points = [tuple(map(int, pt)) for pt in self.all_points]

            n = len(self.all_points)
            if n < 2:
                self.tsp_error_msg = "ERROR: Less than 2 TSP points detected!"
                self.tsp_error_active = True
                self.tsp_error_wait_until = time.time() + TSP_POINT_ERROR_RETRY_SECONDS
                return False
            elif n != CURRENT_EXPECTED_POINTS:
                self.tsp_error_msg = f"ERROR: Detected {n} TSP points, expected {CURRENT_EXPECTED_POINTS}!"
                self.tsp_error_active = True
                self.tsp_error_wait_until = time.time() + TSP_POINT_ERROR_RETRY_SECONDS
                return False
            else:
                self.tsp_error_msg = ""
                self.tsp_error_active = False


            p1 = marker_map[1][0]
            p2 = marker_map[2][0]
            cx1, cy1 = int(np.mean(p1[:, 0])), int(np.mean(p1[:, 1]))
            cx2, cy2 = int(np.mean(p2[:, 0])), int(np.mean(p2[:, 1]))
            pixel_dist = np.linalg.norm(np.array([cx1, cy1]) - np.array([cx2, cy2]))
            self.pixels_per_meter = pixel_dist / 1.3
            return True
        except Exception as e:
            self.pixels_per_meter = 1.0
            print("Calibration failed:", e)
            return False

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
        if not self.visited_points or surf is None:
            return
        
        # Additional safety checks to prevent buffer overflows
        if len(self.visited_points) == 0:
            return
            
        try:
            # Draw path lines between visited points with bounds checking
            for i in range(1, len(self.visited_points)):
                if i < len(self.visited_points) and (i-1) >= 0:
                    pt1 = self.visited_points[i-1]
                    pt2 = self.visited_points[i]
                    # Validate points are within reasonable bounds
                    if (isinstance(pt1, (tuple, list)) and len(pt1) >= 2 and
                        isinstance(pt2, (tuple, list)) and len(pt2) >= 2 and
                        0 <= pt1[0] <= VIDEO_W and 0 <= pt1[1] <= VIDEO_H and
                        0 <= pt2[0] <= VIDEO_W and 0 <= pt2[1] <= VIDEO_H):
                        pygame.draw.line(surf, PATH_COLOR, pt1, pt2, 4)
                        
            # Draw visited point circles with bounds checking
            for pt in self.visited_points:
                if (isinstance(pt, (tuple, list)) and len(pt) >= 2 and
                    0 <= pt[0] <= VIDEO_W and 0 <= pt[1] <= VIDEO_H):
                    pygame.draw.circle(surf, VISITED_PT_COLOR, (int(pt[0]), int(pt[1])), 8)
                    
            # Draw current path and id10 marker
            if not self.path_complete:
                if (len(self.visited_points) > 0 and id10_pos is not None and
                    isinstance(id10_pos, (tuple, list)) and len(id10_pos) >= 2 and
                    0 <= id10_pos[0] <= VIDEO_W and 0 <= id10_pos[1] <= VIDEO_H):
                    
                    last_pt = self.visited_points[-1]
                    if (isinstance(last_pt, (tuple, list)) and len(last_pt) >= 2 and
                        0 <= last_pt[0] <= VIDEO_W and 0 <= last_pt[1] <= VIDEO_H):
                        pygame.draw.line(surf, PATH_COLOR, last_pt, id10_pos, 4)
                    
                    if self.returning_to_start and len(self.visited_points) > 0:
                        first_pt = self.visited_points[0]
                        if (isinstance(first_pt, (tuple, list)) and len(first_pt) >= 2 and
                            0 <= first_pt[0] <= VIDEO_W and 0 <= first_pt[1] <= VIDEO_H):
                            pygame.draw.line(surf, CLOSING_LINE_COLOR, id10_pos, first_pt, 4)
                    
                    pygame.draw.circle(surf, ID10_COLOR, (int(id10_pos[0]), int(id10_pos[1])), 12)
            else:
                # Draw closing line for completed path
                if len(self.visited_points) > 1:
                    last_pt = self.visited_points[-1]
                    first_pt = self.visited_points[0]
                    if (isinstance(last_pt, (tuple, list)) and len(last_pt) >= 2 and
                        isinstance(first_pt, (tuple, list)) and len(first_pt) >= 2 and
                        0 <= last_pt[0] <= VIDEO_W and 0 <= last_pt[1] <= VIDEO_H and
                        0 <= first_pt[0] <= VIDEO_W and 0 <= first_pt[1] <= VIDEO_H):
                        pygame.draw.line(surf, PATH_COLOR, last_pt, first_pt, 4)
                        
        except (IndexError, TypeError, ValueError) as e:
            print(f"Draw path error (safely handled): {e}")
            # Continue without drawing rather than crashing

    def draw_solution(self, surf):
        if not self.show_solution or not self.optimal_path_ready or not self.optimal_path or surf is None:
            return
            
        try:
            # Draw green lines for solution path with bounds checking
            for i in range(1, len(self.optimal_path)):
                if i < len(self.optimal_path) and (i-1) >= 0:
                    pt1 = self.optimal_path[i-1]
                    pt2 = self.optimal_path[i]
                    if (isinstance(pt1, (tuple, list)) and len(pt1) >= 2 and
                        isinstance(pt2, (tuple, list)) and len(pt2) >= 2 and
                        0 <= pt1[0] <= VIDEO_W and 0 <= pt1[1] <= VIDEO_H and
                        0 <= pt2[0] <= VIDEO_W and 0 <= pt2[1] <= VIDEO_H):
                        pygame.draw.line(surf, SOLUTION_COLOR,
                                         tuple(map(int, pt1)),
                                         tuple(map(int, pt2)), 4)
                                         
            # Draw closing line with bounds checking
            if len(self.optimal_path) > 1:
                last_pt = self.optimal_path[-1]
                first_pt = self.optimal_path[0]
                if (isinstance(last_pt, (tuple, list)) and len(last_pt) >= 2 and
                    isinstance(first_pt, (tuple, list)) and len(first_pt) >= 2 and
                    0 <= last_pt[0] <= VIDEO_W and 0 <= last_pt[1] <= VIDEO_H and
                    0 <= first_pt[0] <= VIDEO_W and 0 <= first_pt[1] <= VIDEO_H):
                    pygame.draw.line(surf, SOLUTION_COLOR,
                                     tuple(map(int, last_pt)),
                                     tuple(map(int, first_pt)), 4)

        except (IndexError, TypeError, ValueError) as e:
            print(f"Draw solution error (safely handled): {e}")
            # Continue without drawing rather than crashing

        # Draw red points for TSP points with bounds checking
        if self.all_points:
            try:
                for pt in self.all_points:
                    if (isinstance(pt, (tuple, list)) and len(pt) >= 2 and
                        0 <= pt[0] <= VIDEO_W and 0 <= pt[1] <= VIDEO_H):
                        pygame.draw.circle(surf, SOLUTION_PT_COLOR, tuple(map(int, pt)), 6)
            except (IndexError, TypeError, ValueError) as e:
                print(f"Draw solution points error (safely handled): {e}")
        for x, y in self.optimal_path:
            pygame.draw.circle(surf, SOLUTION_PT_COLOR, (int(x), int(y)), 8)

    def compute_total_distance(self):
        if self.pixels_per_meter is None or len(self.visited_points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.visited_points)):
            total += np.linalg.norm(np.array(self.visited_points[i]) - np.array(self.visited_points[i-1]))

        if self.path_complete and len(self.visited_points) > 1:
            total += np.linalg.norm(np.array(self.visited_points[0]) - np.array(self.visited_points[-1]))
        return total / self.pixels_per_meter

    def compute_optimal_path_distance(self):

        if not self.optimal_path or len(self.optimal_path) < 2 or not self.pixels_per_meter:
            return 0.0
        total = 0.0
        for i in range(1, len(self.optimal_path)):
            total += np.linalg.norm(np.array(self.optimal_path[i]) - np.array(self.optimal_path[i-1]))
        # Closing the loop:

        total += np.linalg.norm(np.array(self.optimal_path[0]) - np.array(self.optimal_path[-1]))
        return total / self.pixels_per_meter

    def capture_loop(self):
        if not self.picam2:
            print("Camera not initialized properly")
            self.running = False
            return
        
        # Add sleep after thread start to mimic gdb behavior
        sleep(0.2)
        print("Capture thread started with safety delay")
        
        initialized_points = False

        try:
            while self.running:
                # Check for restart commands from main thread
                try:
                    restart_cmd = self.restart_queue.get_nowait()
                    if restart_cmd:
                        print("Restart command received in capture thread")
                        initialized_points = False
                        # Clear trail safely
                        with self.trail_lock:
                            self.trail.clear()
                except queue.Empty:
                    pass
                
                # Capture frame with proper error handling
                try:
                    frame = self.picam2.capture_array("main")
                    if frame is None:
                        print("Warning: Received null frame from camera")
                        sleep(0.1)
                        continue
                    
                    # Create defensive copy to prevent use-after-free
                    frame_copy = frame.copy()
                except Exception as e:
                    print(f"Error capturing frame: {e}")
                    sleep(0.1)
                    continue
                
                if not initialized_points:
                    now = time.time()
                    if self.corners_error_active:
                        if now >= self.corners_error_wait_until:
                            if not self.initialize_points_and_calibration(frame_copy):
                                self.corners_error_wait_until = now + CORNERS_ERROR_RETRY_SECONDS
                                # Send frame to main thread via queue
                                try:
                                    self.frame_queue.put(frame_copy, block=False)
                                except queue.Full:
                                    # If queue is full, remove old frame and add new one
                                    try:
                                        self.frame_queue.get_nowait()
                                        self.frame_queue.put(frame_copy, block=False)
                                    except queue.Empty:
                                        pass
                                sleep(0.1)
                                continue
                            else:
                                self.corners_error_active = False
                        else:
                            try:
                                self.frame_queue.put(frame_copy, block=False)
                            except queue.Full:
                                try:
                                    self.frame_queue.get_nowait()
                                    self.frame_queue.put(frame_copy, block=False)
                                except queue.Empty:
                                    pass
                            sleep(0.1)
                            continue
                    elif self.tsp_error_active:
                        if now >= self.tsp_error_wait_until:
                            if not self.initialize_points_and_calibration(frame_copy):
                                self.tsp_error_wait_until = now + TSP_POINT_ERROR_RETRY_SECONDS
                                try:
                                    self.frame_queue.put(frame_copy, block=False)
                                except queue.Full:
                                    try:
                                        self.frame_queue.get_nowait()
                                        self.frame_queue.put(frame_copy, block=False)
                                    except queue.Empty:
                                        pass
                                sleep(0.1)
                                continue
                            else:
                                self.tsp_error_active = False
                                self.tsp_error_msg = ""
                                if self.all_points:
                                    opt_path, _ = compute_tsp_with_convex_hull(self.all_points)
                                    self.optimal_path = [tuple(map(int, pt)) for pt in opt_path]
                                    self.optimal_path_ready = True
                                initialized_points = True
                        else:
                            try:
                                self.frame_queue.put(frame_copy, block=False)
                            except queue.Full:
                                try:
                                    self.frame_queue.get_nowait()
                                    self.frame_queue.put(frame_copy, block=False)
                                except queue.Empty:
                                    pass
                            sleep(0.1)
                            continue
                    else:
                        if self.initialize_points_and_calibration(frame_copy):
                            if self.all_points:
                                opt_path, _ = compute_tsp_with_convex_hull(self.all_points)
                                self.optimal_path = [tuple(map(int, pt)) for pt in opt_path]
                                self.optimal_path_ready = True
                            initialized_points = True
                        else:
                            try:
                                self.frame_queue.put(frame_copy, block=False)
                            except queue.Full:
                                try:
                                    self.frame_queue.get_nowait()
                                    self.frame_queue.put(frame_copy, block=False)
                                except queue.Empty:
                                    pass
                            sleep(0.1)
                            continue

                # Process markers and trail updates
                marker_map, _ = detect_aruco_markers(frame_copy)
                id10_pos = None
                if config.MOVING_ID in marker_map:
                    pts = marker_map[config.MOVING_ID][0]
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    id10_pos = (cx, cy)
                    
                    # Update trail safely and send update to main thread
                    if not self.path_complete:
                        with self.trail_lock:
                            try:
                                self.trail.append(id10_pos)
                                # Send trail update to main thread
                                self.trail_update_queue.put({'action': 'add', 'pos': id10_pos}, block=False)
                            except queue.Full:
                                # If queue is full, just continue - we'll get the next position
                                pass
                else:
                    # Send frame even if no marker detected
                    try:
                        self.frame_queue.put(frame_copy, block=False)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame_copy, block=False)
                        except queue.Empty:
                            pass
                    sleep(0.1)
                    continue

                # Update path logic
                if not self.path_complete and id10_pos is not None:
                    self.update_path_logic(id10_pos)

                # Send frame to main thread via queue
                try:
                    self.frame_queue.put(frame_copy, block=False)
                except queue.Full:
                    # Replace old frame with new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame_copy, block=False)
                    except queue.Empty:
                        pass

                # Add small delay to prevent excessive CPU usage and mimic gdb behavior
                sleep(0.1)
                
        except Exception as e:
            print(f"Critical error in capture loop: {e}")
            self.running = False
        finally:
            # Safe camera cleanup
            if self.picam2:
                try:
                    print("Shutting down camera...")
                    self.picam2.stop()
                    sleep(0.2)  # Allow camera to properly shut down
                except Exception as e:
                    print(f"Error stopping camera: {e}")
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
        if self.show_solution:
            opt_dist_m = self.compute_optimal_path_distance()
            opt_dist_str = f"Optimal Path: {opt_dist_m:.2f} m"
            opt_dist_label = self.info_font.render(opt_dist_str, True, (20, 100, 20))
            self.screen.blit(opt_dist_label, (VIDEO_W + BUTTON_PAD, y))
            y += linespacing

        def draw_wrapped_text(text, font, color, x, y, max_width):
            words = text.split(' ')
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + (' ' if current_line else '') + word
                test_surface = font.render(test_line, True, color)
                if test_surface.get_width() <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            for line in lines:
                line_surface = font.render(line, True, color)
                self.screen.blit(line_surface, (x, y))
                y += line_surface.get_height() + 2
            return y

        if self.tsp_error_msg and self.tsp_error_active:
            timer_left = int(self.tsp_error_wait_until - time.time())
            if timer_left < 0:
                timer_left = 0
            err_text = self.tsp_error_msg + f" | Retrying in {timer_left} seconds"
            y = draw_wrapped_text(
                err_text, self.error_font, (255, 0, 0),
                VIDEO_W + BUTTON_PAD, y, WIN_W - VIDEO_W - 2 * BUTTON_PAD
            )
            y += linespacing

        if self.corners_error_msg and self.corners_error_active:
            timer_left = int(self.corners_error_wait_until - time.time())
            if timer_left < 0:
                timer_left = 0
            err_text = self.corners_error_msg + f" | Retrying in {timer_left} seconds"
            y = draw_wrapped_text(
                err_text, self.error_font, (255, 0, 0),
                VIDEO_W + BUTTON_PAD, y, WIN_W - VIDEO_W - 2 * BUTTON_PAD
            )
            y += linespacing

    def display_loop(self):
        self.restart_hovered = False
        self.solution_hovered = False
        current_frame = None

        while True:
            # Process pygame events (main thread only)
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

            # Process trail updates from capture thread
            try:
                while True:
                    trail_update = self.trail_update_queue.get_nowait()
                    if trail_update['action'] == 'add':
                        # Trail updates are already handled in capture thread with locks
                        # This is just for potential future use
                        pass
            except queue.Empty:
                pass

            # Get latest frame from capture thread via queue
            try:
                while True:
                    # Get all available frames, keeping only the latest
                    current_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                pass

            # Render GUI (all GUI operations in main thread)
            self.screen.fill((220, 220, 220))

            if current_frame is not None:
                try:
                    # Process frame safely
                    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    surf = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
                    surf = pygame.transform.scale(surf, (VIDEO_W, VIDEO_H))
                    self.video_surf = surf
                except Exception as e:
                    print(f"Error processing frame for display: {e}")
                    # Continue with previous frame if available

            if self.video_surf:
                self.screen.blit(self.video_surf, (0, 0))

                overlay = pygame.Surface((VIDEO_W, VIDEO_H), pygame.SRCALPHA)
                if self.visited_points:
                    # Safely determine the current position for drawing
                    current_pos = None
                    with self.trail_lock:
                        try:
                            if len(self.trail) > 0:
                                current_pos = self.trail[-1]
                        except (IndexError, RuntimeError) as e:
                            # Handle any race conditions gracefully
                            print(f"Trail access warning: {e}")
                            pass
                    
                    if current_pos is None and len(self.visited_points) > 0:
                        current_pos = self.visited_points[-1]
                    
                    if current_pos is not None:
                        self.draw_path(overlay, current_pos)
                
                self.draw_solution(overlay)
                self.screen.blit(overlay, (0, 0))

            # Draw UI elements (all in main thread)
            self.draw_info()
            self.draw_buttons()

            # Update display
            pygame.display.flip()
            self.clock.tick(30)
            
            if not self.running:
                break

        # Clean shutdown
        pygame.quit()

    def restart(self):
        print("Restart initiated from main thread")
        # Send restart command to capture thread via queue
        try:
            self.restart_queue.put(True, block=False)
        except queue.Full:
            # Clear queue and send new restart command
            try:
                self.restart_queue.get_nowait()
            except queue.Empty:
                pass
            self.restart_queue.put(True, block=False)
        
        # Clear trail safely in main thread
        with self.trail_lock:
            self.trail.clear()
        
        # Reset state variables (main thread only)
        self.visited_points = []
        self.path_complete = False
        self.returning_to_start = False
        self.show_solution = False
        self.tsp_error_msg = ""
        self.tsp_error_active = False
        self.corners_error_msg = ""
        self.corners_error_active = False
        
        print("Restart completed")


    def run(self):
        print("Starting ArUco tracker with Pygame UI...")

        # Initialize camera in main thread to avoid segmentation faults
        try:
            print("Initializing camera...")
            self.picam2 = Picamera2()
            
            # Add delay to mimic gdb behavior and allow proper initialization
            sleep(0.2)
            
            config_cam = self.picam2.create_preview_configuration(main={"size": (VIDEO_W, VIDEO_H)})
            self.picam2.configure(config_cam)
            
            # Add delay after configuration
            sleep(0.2)
            
            self.picam2.start()
            
            # Extended delay after camera start to ensure stability
            sleep(1.0)
            print("Camera initialized successfully")
            
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            if self.picam2:
                try:
                    self.picam2.stop()
                    sleep(0.2)  # Allow proper cleanup
                except Exception as cleanup_error:
                    print(f"Error during camera cleanup: {cleanup_error}")
            return

        # Add delay before starting threads to mimic gdb behavior
        sleep(0.2)
        print("Starting capture thread...")
        
        try:
            capture_thread = threading.Thread(target=self.capture_loop, name="CaptureThread")
            capture_thread.daemon = True
            capture_thread.start()
            
            # Give capture thread time to start
            sleep(0.2)
            print("Capture thread started, beginning display loop")

            # Run display loop in main thread (GUI operations must be in main thread)
            self.display_loop()

        except Exception as e:
            print(f"Error during thread operations: {e}")
            self.running = False
        finally:
            # Clean shutdown
            print("Shutting down threads...")
            self.running = False
            
            if 'capture_thread' in locals():
                print("Waiting for capture thread to finish...")
                capture_thread.join(timeout=3)
                if capture_thread.is_alive():
                    print("Warning: Capture thread did not shut down cleanly")
            
            print("Tracker shutdown complete")

if __name__ == "__main__":
    show_menu_and_get_expected_points()
    tracker = ArucoTracker()
    tracker.run()