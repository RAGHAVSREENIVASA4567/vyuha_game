# AETHON IMU Stream

This folder contains the AETHON IMU telemetry tools and ESP32 firmware:

- `aethon_esp32.ino` and `final/final.ino`: Arduino firmware variants.
- `litewing_mpu_bridge.py`: MPU/serial telemetry bridge.
- `aethon_visualizer_3d.py`: Panda3D telemetry visualizer.
- `hand_tracker.py`: hand-tracking input support.
- `drone.glb` and `hand_landmarker.task`: bundled model assets.

Run the Python tools with their required Python packages installed. Keep virtual
environments and generated caches out of version control; the included
`.gitignore` covers common local artifacts.
