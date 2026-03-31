import sys
import os
from src.utils.logger import logger
from config.settings import CONFIG

"""
Main Application Entry Point
----------------------------
Orchestrates the startup, configuration loading, camera stream,
and main loop for the Face Recognition Attendance System.
"""

def main():
    """
    Bootstrap function for the entire system.
    Steps:
        1. Load and validate configurations.
        2. Initialize the logger.
        3. Establish database connection.
        4. Initialize the camera/video stream.
        5. Enter the recognition and attendance marking loop.
    """
    logger.info("Starting Face Recognition Attendance System...")
    logger.info(f"Using Configuration from file config/config.yaml")

    camera_source = CONFIG.get('camera', {}).get('source', 0)
    logger.info(f"Attempting to initialize camera source: {camera_source}")

    # Core loop (to be implemented)
    # 1. Capture Frame from WebCam
    # 2. Detect Faces
    # 3. Identify Faces
    # 4. Mark Attendance
    # 5. Display Result / Dashboard

    logger.info("System initialized successfully. Entering capture loop.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("System shutdown requested by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical system failure: {str(e)}")
        sys.exit(1)
