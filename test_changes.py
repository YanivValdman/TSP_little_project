#!/usr/bin/env python3
"""
Test script to validate our changes without full GUI
"""

def test_mock_marker_detection():
    """Test that mock marker detection works"""
    from utils import detect_aruco_markers, _is_mock_image, _detect_mock_markers
    import numpy as np
    import cv2
    import time
    
    # Create a mock image similar to what MockCamera generates
    img = np.full((600, 800, 3), 128, dtype=np.uint8)  # Gray background
    
    # Add some colored rectangles to simulate ArUco markers
    cv2.rectangle(img, (50, 50), (100, 100), (0, 0, 255), -1)    # Corner 1
    cv2.rectangle(img, (700, 50), (750, 100), (0, 0, 255), -1)   # Corner 2
    cv2.rectangle(img, (50, 500), (100, 550), (0, 0, 255), -1)   # Corner 3
    cv2.rectangle(img, (700, 500), (750, 550), (0, 0, 255), -1)  # Corner 4
    
    # TSP points (blue squares) - non-colinear
    cv2.rectangle(img, (200, 200), (230, 230), (255, 0, 0), -1)  # TSP point 1
    cv2.rectangle(img, (450, 150), (480, 180), (255, 0, 0), -1)  # TSP point 2
    cv2.rectangle(img, (550, 350), (580, 380), (255, 0, 0), -1)  # TSP point 3
    cv2.rectangle(img, (250, 450), (280, 480), (255, 0, 0), -1)  # TSP point 4
    cv2.rectangle(img, (600, 200), (630, 230), (255, 0, 0), -1)  # TSP point 5
    
    # Moving marker (green square)
    cv2.rectangle(img, (350, 300), (380, 330), (0, 255, 0), -1)   # Moving marker (ID 10)
    
    print("Testing mock image detection...")
    print(f"Is mock image: {_is_mock_image(img)}")
    
    marker_map, corners = detect_aruco_markers(img)
    print(f"Detected {len(marker_map)} markers")
    print(f"Marker IDs: {list(marker_map.keys())}")
    
    # Test corner detection
    corner_ids = [1, 2, 3, 4]
    corners_found = all(id_ in marker_map for id_ in corner_ids)
    print(f"All corners found: {corners_found}")
    
    return len(marker_map) >= 9  # Should have 4 corners + 5 TSP points + 1 moving

def test_tsp_with_mock_points():
    """Test TSP computation with mock points"""
    from heuristic_tsp import compute_tsp_with_convex_hull
    
    # Mock TSP points (non-colinear)
    points = [
        (215, 215),  # TSP point 1
        (465, 165),  # TSP point 2  
        (565, 365),  # TSP point 3
        (265, 465),  # TSP point 4
        (615, 215),  # TSP point 5
    ]
    
    print("\nTesting TSP computation...")
    print(f"Input points: {points}")
    
    try:
        path, cost = compute_tsp_with_convex_hull(points)
        print(f"TSP path computed successfully")
        print(f"Path length: {len(path)}")
        print(f"Total cost: {cost:.2f}")
        return True
    except Exception as e:
        print(f"TSP computation failed: {e}")
        return False

def test_fullscreen_config():
    """Test that fullscreen configuration would work"""
    import pygame
    
    print("\nTesting pygame fullscreen configuration...")
    try:
        pygame.init()
        # Don't actually create fullscreen, just test the setup
        info = pygame.display.Info()
        print(f"Screen size would be: {info.current_w} x {info.current_h}")
        pygame.quit()
        return True
    except Exception as e:
        print(f"Pygame setup failed: {e}")
        return False

if __name__ == "__main__":
    print("Running validation tests for TSP changes...")
    
    test1_pass = test_mock_marker_detection()
    test2_pass = test_tsp_with_mock_points()
    test3_pass = test_fullscreen_config()
    
    print(f"\nTest Results:")
    print(f"Mock marker detection: {'PASS' if test1_pass else 'FAIL'}")
    print(f"TSP computation: {'PASS' if test2_pass else 'FAIL'}")
    print(f"Fullscreen config: {'PASS' if test3_pass else 'FAIL'}")
    
    if all([test1_pass, test2_pass, test3_pass]):
        print("\nAll tests PASSED! ✓")
    else:
        print("\nSome tests FAILED! ✗")