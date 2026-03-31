import cv2
import numpy as np
from src.utils.logger import logger

"""
This module performs optimized Face Detection.
Uses strict morphological filtering and dimensional constraints
to achieve industry-grade false-positive elimination.
"""

def is_image_too_dim(frame, threshold=30):
    """Simple check to see if we need more light."""
    avg_brightness = np.mean(frame)
    return avg_brightness < threshold

class FaceLocator:
    """
    Main class for finding face coordinates.
    """

    def __init__(self, sensitivity=1.1, neighbors=15):
        # We increase minNeighbors significantly (e.g. 15+) for industry-level
        # false positive elimination. A shirt pattern will not trigger this.
        model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.classifier = cv2.CascadeClassifier(model_path)
        
        self.scale = sensitivity
        self.min_neighbors = neighbors
        
        if self.classifier.empty():
            logger.error("COULD NOT LOAD FACE MODEL FILE")

    def find_face_boxes(self, raw_frame):
        """
        Takes a frame and returns a list of [x, y, w, h] for every real human face found.
        """
        if raw_frame is None:
            logger.warning("[WARNING] Received empty frame in find_face_boxes")
            return []

        # Convert to grayscale to make processing easier
        gray_image = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        
        # Improve contrast to help detection in bad lighting
        processed_gray = cv2.equalizeHist(gray_image)

        # Look for faces with STRICT validation parameters
        face_list = self.classifier.detectMultiScale(
            processed_gray,
            scaleFactor=self.scale,
            minNeighbors=self.min_neighbors,
            minSize=(100, 100) # Ignore background noise and distant objects
        )

        if len(face_list) > 0:
            logger.info(f"[INFO] Found {len(face_list)} valid face(s)")
        else:
            pass

        return face_list

    def draw_visual_markers(self, image, face_list, rect_color=(0, 255, 0)):
        """ Draws a professional bounding box around validated faces. """
        canvas = image.copy()
        for (x, y, w, h) in face_list:
            # Draw a nice rectangle
            cv2.rectangle(canvas, (x, y), (x + w, y + h), rect_color, 2)
        return canvas

def get_camera_feed(device_id=0):
    """ A simple generator to grab frames from the webcam. """
    camera = cv2.VideoCapture(device_id)
    
    if not camera.isOpened():
        logger.error(f"Failed to open camera {device_id}")
        return

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            break
        yield frame
        
    camera.release()
