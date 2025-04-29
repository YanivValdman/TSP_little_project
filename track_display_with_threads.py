import cv2
import numpy as np
import pygame
from time import sleep
import threading
from collections import deque
from picamera2 import Picamera2
from utils import detect_aruco_markers

# === CONFIG ===
MOVING_ID = 10             # The marker being tracked
MAX_TRAIL = 1000           # Maximum trail length to display

class ArucoTracker:
    def __init__(self):
        self.running = True
        self.trail = deque(maxlen=MAX_TRAIL)
        self.lock = threading.Lock()
        self.current_frame = None
        self.frame_ready = False
        
        # Initialize pygame (instead of OpenCV GUI)
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("ArUco Tracker")
    
    def draw_trail(self, image, trail):
        """Draw the movement trail of the tracked marker"""
        for i in range(1, len(trail)):
            cv2.line(image, trail[i-1], trail[i], (0, 0, 255), 2)
        if trail:
            cv2.circle(image, trail[-1], 5, (0, 0, 255), -1)
    
    def capture_loop(self):
        """Thread function to capture and process frames"""
        # Initialize camera
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (800, 600)})
        picam2.configure(config)
        picam2.start()
        sleep(1)
        
        count = 0
        try:
            while self.running:
                count += 1
                print(f"{count=}")
                
                # Capture and process frame
                frame = picam2.capture_array("main")
                marker_map, _ = detect_aruco_markers(frame)
                
                # Track the moving marker
                if MOVING_ID in marker_map:
                    pts = marker_map[MOVING_ID][0]
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    self.trail.append((cx, cy))
                
                # Create a display frame with trail
                display_frame = frame.copy()
                self.draw_trail(display_frame, self.trail)
                
                # Update the current frame with lock to avoid race conditions
                with self.lock:
                    self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    self.frame_ready = True
                
                # Slow down capture rate
                sleep(0.1)
        finally:
            picam2.stop()
            print("Camera resources released")
    
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