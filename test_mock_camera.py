#!/usr/bin/env python3
"""
Test script for TSP tracker with mock camera for systems without Picamera2
"""
import cv2
import numpy as np
import time
from time import sleep
import threading
from unittest.mock import MagicMock
import sys

# Mock Picamera2 for testing
class MockPicamera2:
    def __init__(self):
        self.running = False
        
    def create_preview_configuration(self, main=None):
        return {}
        
    def configure(self, config):
        pass
        
    def start(self):
        self.running = True
        
    def stop(self):
        self.running = False
        
    def capture_array(self, mode):
        # Generate a test image with some colored squares as mock ArUco markers
        img = np.full((600, 800, 3), 128, dtype=np.uint8)  # Gray background
        
        # Add some colored rectangles to simulate ArUco markers
        # Corner markers (red squares)
        cv2.rectangle(img, (50, 50), (100, 100), (0, 0, 255), -1)    # Top-left (ID 1)
        cv2.rectangle(img, (700, 50), (750, 100), (0, 0, 255), -1)   # Top-right (ID 2)
        cv2.rectangle(img, (50, 500), (100, 550), (0, 0, 255), -1)   # Bottom-left (ID 3)
        cv2.rectangle(img, (700, 500), (750, 550), (0, 0, 255), -1)  # Bottom-right (ID 4)
        
        # TSP points (blue squares)
        cv2.rectangle(img, (200, 200), (230, 230), (255, 0, 0), -1)  # TSP point 1
        cv2.rectangle(img, (400, 150), (430, 180), (255, 0, 0), -1)  # TSP point 2
        cv2.rectangle(img, (500, 300), (530, 330), (255, 0, 0), -1)  # TSP point 3
        cv2.rectangle(img, (300, 400), (330, 430), (255, 0, 0), -1)  # TSP point 4
        cv2.rectangle(img, (600, 250), (630, 280), (255, 0, 0), -1)  # TSP point 5
        
        # Moving marker (green square) - simulates robot position
        # Make it move slowly in a pattern
        t = time.time() * 0.5
        x = int(350 + 100 * np.sin(t))
        y = int(300 + 50 * np.cos(t))
        cv2.rectangle(img, (x, y), (x+30, y+30), (0, 255, 0), -1)   # Moving marker (ID 10)
        
        return img

# Monkey patch the import
sys.modules['picamera2'] = MagicMock()
sys.modules['picamera2'].Picamera2 = MockPicamera2

# Now import and run the tracker
from track_display_with_threads import show_menu_and_get_expected_points, ArucoTracker

if __name__ == "__main__":
    print("Starting TSP Tracker with mock camera...")
    show_menu_and_get_expected_points()
    tracker = ArucoTracker()
    tracker.run()