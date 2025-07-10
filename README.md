# Real-time TSP Solver and Visualizer (Raspberry Pi + ArUco Markers)

This project is a **real-time TSP (Traveling Salesman Problem) solver and visualizer** using a camera and ArUco markers. It is designed to run on a dedicated Raspberry Pi and automatically starts on system boot. The system detects ArUco markers in a camera feed, tracks a moving object, and visualizes both the optimal and live traversed paths over a set of detected points.

---

## Features

- **Startup Menu**: On boot or script start, select between a default or custom number of TSP points (2-30), with keyboard navigation and timeout.
- **Real-Time Computer Vision**: Uses OpenCV and ArUco markers to detect a set of points (targets) and a moving object in the camera feed.
- **TSP Path Calculation**: Computes an approximate optimal route using a convex hull heuristic.
- **Live Path Tracking**: Tracks and visualizes the path taken by the moving object (e.g., robot or vehicle) in real time.
- **UI Display**: Pygame-based display panel with:
  - Live video stream
  - Progress and statistics (distance, points visited, solution length)
  - Error messages (always wrapped and visible)
  - Interactive buttons for restart and toggling the solution display
- **Robust Error Handling**: Timed error displays and retry logic for marker or corner detection.

---

## ArUco Markers (aruco.pdf)

The repository includes a file called `aruco.pdf`, which contains printable ArUco markers used for detection:

- **Markers 1-4:**  
  These should be printed and placed at the **corners** of the workspace. They are used to calibrate and warp the camera image so all detected points are mapped correctly.
- **Marker 10:**  
  This is used as the **default moving marker**. Attach this to the robot or moving object you want to track.
- **Other Markers:**  
  Additional ArUco markers from the PDF can be used as the TSP points (targets to visit).

**Print `aruco.pdf` at high quality and cut out the markers you need for your setup.**

---

## Quick Start

### **Requirements**

- **Hardware**: Raspberry Pi with a supported camera.
- **OS**: Raspberry Pi OS (or another Linux distro capable of running OpenCV, Pygame, Picamera2).
- **Python**: Python 3.7+ recommended.
- **Dependencies**:
  - `opencv-python`
  - `numpy`
  - `pygame`
  - `picamera2`
  - `matplotlib`
  - `scipy`

Install dependencies via:

```bash
pip install opencv-python numpy pygame picamera2 matplotlib scipy
```

### **Configuration**

Edit `config.py` to set your ArUco marker setup:

```python
# Configuration module for tracker settings

# The ArUco marker ID to track as the moving point
MOVING_ID = 10

# The expected number of TSP points (targets to visit)
EXPECTED_TSP_POINTS = 5  # Change this to your setup
```

> **Note:** You can override the number of points at startup via the menu.

---

## Running

### **On Raspberry Pi Boot**

You can configure your Raspberry Pi to run this script automatically on startup (e.g., via `rc.local` or a systemd service):

```bash
python3 track_display_with_threads.py
```

Or, simply run manually for testing:

```bash
python3 track_display_with_threads.py
```

On launch, you'll see a menu:
- **Up/Down + Enter** to choose: use default config, or enter a custom number of points.
- **Timeout**: If no selection is made, default configuration is selected after 10 seconds (message is wrapped and always visible).

---

## How It Works

1. **Marker Detection**:
   - Four ArUco markers (IDs 1-4) mark the corners of the field of view.
   - Additional markers are placed as TSP points (targets) and one is tracked as the "moving" point (default ID 10).

2. **Image Warping**:
   - The image is rectified (warped) based on corner marker detection for accurate TSP computation.

3. **TSP Path Calculation**:
   - All "target" markers (excluding corners and the moving marker) are detected.
   - A convex hull-based heuristic is used to quickly compute an approximate optimal tour.

4. **Live Tracking and UI**:
   - The moving marker's position is tracked in real time.
   - The path taken is displayed, along with the optimal path for comparison, and statistics.
   - Errors (e.g., missing markers) are shown with auto-retry and clear feedback.

---

## Main Files and Functions

- **track_display_with_threads.py**  
  Main script: GUI, camera capture, path logic, and application loop.

- **config.py**  
  Marker configuration (see above).

- **utils.py**  
  - `detect_aruco_markers(image)`: Finds all ArUco markers in the image.
  - `detect_corners_and_warp(image)`: Detects corner markers and warps the image for TSP computation.
  - Additional image and plotting utilities.

- **heuristic_tsp.py**  
  - `compute_tsp_with_convex_hull(points)`: Fast TSP approximation using convex hull insertion.
  - `detect_tsp_points_in_warped_image(warped_image, car_id)`: Finds all TSP target marker centers in the warped image.

- **aruco.pdf**  
  Printable ArUco markers for your experiment.

---

## Authors

- Rashel Strigevsky
- Yaniv Valdman

---

## License

No license specified.  
If you wish to open source this project, consider adding a license file (e.g., MIT, GPL, etc.).

---

## Acknowledgements

- OpenCV (https://opencv.org/)
- Pygame (https://www.pygame.org/)
- Picamera2 (https://github.com/raspberrypi/picamera2)
- SciPy and Matplotlib for TSP algorithms and plotting


## Project Structure

```
track_display_with_threads.py
config.py
utils.py
heuristic_tsp.py
aruco.pdf
...
```

---
