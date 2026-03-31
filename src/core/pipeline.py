import cv2
import numpy as np
import threading
import queue
import time
from src.utils.logger import logger
from config.settings import CONFIG

from src.core.face_recognition.face_detection import FaceLocator
from src.core.face_recognition.face_recognition import PersonIdentifier
from src.core.face_recognition.liveness import LivenessDetector
from src.database.database import DatabaseManager

class AttendanceSystem:
    def __init__(self):
        # Tools we need
        self.locator = FaceLocator()
        self.identifier = PersonIdentifier()
        self.liveness_check = LivenessDetector()
        self.db = DatabaseManager()
        
        self.people_db = []
        self.load_registered_users()

        # Threaded AI system
        self.recognition_results = {} # Map of face_id -> (name, confidence)
        self.processing_queue = queue.Queue(maxsize=1) # Only keep 1 latest task to avoid lag build-up
        self.is_running = True
        
        self.worker_thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self.worker_thread.start()

        self.match_threshold = CONFIG.get('recognition', {}).get('threshold', 0.4)

    def _recognition_worker(self):
        """ Background thread that handles the heavy lifting without blocking the video feed. """
        while self.is_running:
            try:
                # Wait for a new frame to process
                task = self.processing_queue.get(timeout=1)
                frame, face_boxes = task
                
                new_results = {}
                for i, (x, y, w, h) in enumerate(face_boxes):
                    # Only identify if person wasn't recently identified or to refresh
                    person, confidence = self.identify_person_in_box(frame, (x, y, w, h))
                    if person:
                        new_results[i] = (person['name'], confidence)
                        
                        # Check if we already logged them recently (e.g., within 1 minute)
                        if not self.db.check_recent_entry(person['id'], minutes=1):
                            self.db.mark_attendance(person['id'])
                            logger.info(f"Attendance automatically logged for {person['name']}")
                            
                    else:
                        new_results[i] = ("Unknown", 0)
                
                self.recognition_results = new_results
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker Error: {e}")

    def load_registered_users(self):
        """ Pulls everyone from the database into memory. """
        self.people_db = self.db.get_all_users()
        logger.info(f"System ready with {len(self.people_db)} enrolled users.")

    def handle_video_frame(self, frame):
        """ Real-time logic that returns frames IMMEDIATELY while AI works in background. """
        # 1. Find all faces (Extremely Fast)
        face_boxes = self.locator.find_face_boxes(frame)
        
        # 2. Draw detection boxes (Real-time)
        output_image = self.locator.draw_visual_markers(frame, face_boxes)

        # 3. Offload recognition to background thread if queue is empty
        if len(face_boxes) > 0 and self.processing_queue.empty():
            try:
                self.processing_queue.put_nowait((frame.copy(), face_boxes))
            except queue.Full:
                pass

        # 4. Display the latest available results from the worker thread
        for i, (x, y, w, h) in enumerate(face_boxes):
            if i in self.recognition_results:
                name, confidence = self.recognition_results[i]
                if name != "Unknown":
                    # Draw identified name (from background result)
                    cv2.putText(output_image, f"{name}", (x, y+h+25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (16, 185, 129), 2)
                    # Note: We should probably trigger attendance in the worker thread too
                else:
                    cv2.putText(output_image, "Unknown", (x, y+h+25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # While waiting for AI, show 'Scanning'
                cv2.putText(output_image, "IDENTIFYING...", (x, y+h+25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1)

        if len(face_boxes) == 0:
            self.recognition_results = {}

        return output_image

    def identify_person_in_box(self, frame, box):
        """ Helper to crop a face and run it through the AI. """
        x, y, w, h = box
        face_crop = frame[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            logger.error("[ERROR] Attempted to identify person with empty crop")
            return None, 0
            
        fingerprint = self.identifier.extract_face_fingerprint(face_crop)
        return self.identifier.find_matching_user(fingerprint, self.people_db, cutoff=self.match_threshold)

    def record_attendance(self, person, image, box, score):
        """ Helper to log the person and update the visual display. """
        x, y, w, h = box
        name = person['name']
        user_id = person['id']
        
        # Draw name on screen
        cv2.putText(image, f"{name} ({score:.0f}%)", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Log to DB if they haven't checked in recently
        if not self.db.check_recent_entry(user_id, minutes=60):
            logger.info(f"[INFO] Registering attendance for {name} in database...")
            if self.db.mark_attendance(user_id):
                logger.info(f"[SUCCESS] Attendance successfully logged for {name}")
            else:
                logger.error(f"[ERROR] Database failed to record attendance for {name}")
        else:
            logger.debug(f"[DEBUG] Skipping DB log for {name} (Already marked recently)")
