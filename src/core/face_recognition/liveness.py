import cv2
import numpy as np
from src.utils.logger import logger
from typing import List, Tuple

"""
Liveness Detection Module
--------------------------
Provides lightweight anti-spoofing checks to distinguish between
a live person and a static photograph.
"""

class LivenessDetector:
    """
    Implements motion-based liveness detection by tracking facial 
    landmarks or bounding box variance across frames.
    """

    def __init__(self, history_size: int = 10, movement_threshold: float = 1.5):
        """
        Initializes the tracker.
        Args:
            history_size (int): Number of frames to track for movement.
            movement_threshold (float): Minimum variance required to consider 'live'.
        """
        self.history_size = history_size
        self.movement_threshold = movement_threshold
        self.center_history = []
        logger.info("LivenessDetector initialized (Motion-based).")

    def check_liveness(self, face_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Calculates if the face shows micro-movements characteristic of a live human.
        
        Args:
            face_bbox (Tuple): (x, y, w, h) of the detected face.
            
        Returns:
            bool: True if motion is detected, False if static or insufficient data.
        """
        if face_bbox is None:
            self.center_history = []
            return False

        x, y, w, h = face_bbox
        center = (x + w // 2, y + h // 2)
        
        self.center_history.append(center)
        
        if len(self.center_history) > self.history_size:
            self.center_history.pop(0)

        if len(self.center_history) < self.history_size:
            # Not enough frames yet
            return False

        # Calculate Variance of centers
        centers_array = np.array(self.center_history)
        variance = np.var(centers_array, axis=0) # [var_x, var_y]
        total_variance = np.sum(variance)

        # Logic: 
        # 1. A perfectly static image (photo on a stand) will have ~0 variance.
        # 2. A live person has micro-movements (breathing, small head shifts).
        # 3. Too much variance might mean the face is moving too fast for recognition.
        
        is_live = total_variance > self.movement_threshold
        
        if is_live:
            logger.debug(f"Liveness detected. Variance: {total_variance:.2f}")
        
        return is_live

    def reset(self):
        """Resets the history for a new detection session."""
        self.center_history = []
