import numpy as np
from deepface import DeepFace
from src.utils.logger import logger

"""
This part of the code handles identity.
It turns a picture of a face into a long list of numbers (a vector).
If two lists of numbers are similar, it means the faces are the same person.
"""

class PersonIdentifier:
    """
    Class to help identify people by comparing their facial fingerprints.
    """

    def __init__(self, ai_model="Facenet"):
        # We're using Facenet because it's accurate and fast
        self.model_choice = ai_model
        logger.info(f"Identity system ready using: {self.model_choice}")

    def extract_face_fingerprint(self, face_image):
        """
        Creates a list of numbers that represents this specific face.
        """
        try:
            logger.debug("[DEBUG] Transforming face crop into numerical fingerprint...")
            # We tell DeepFace to skip its own detection because we manually did it already
            raw_output = DeepFace.represent(
                img_path=face_image,
                model_name=self.model_choice,
                enforce_detection=False,
                detector_backend='skip'
            )
            
            if len(raw_output) == 0:
                logger.warning("[WARNING] DeepFace was unable to extract a feature vector.")
                return None

            signature = np.array(raw_output[0]["embedding"])

            # Normalize the numbers to make comparison mathematically easier
            unit_signature = signature / np.linalg.norm(signature)
            logger.debug("[DEBUG] Vector fingerprint successfully generated and normalized.")
            return unit_signature

        except Exception as err:
            logger.error(f"[ERROR] Failed to extract face signature: {err}")
            return None

    def find_matching_user(self, current_signature, known_people, cutoff=0.4):
        """
        Checks our signature against everyone we know in the database.
        """
        if not known_people:
            logger.warning("[WARNING] No known people to search against. Database might be empty.")
            return None, 0.0

        if current_signature is None:
            return None, 0.0

        best_score = 1.0 # Smaller is better in cosine distance
        matched_person = None

        logger.debug(f"[DEBUG] Comparing face against {len(known_people)} known signatures...")
        for person in known_people:
            # Calculate how 'different' the two faces are
            difference = self.calculate_difference(current_signature, person['embedding'])
            
            if difference < best_score:
                best_score = difference
                matched_person = person

        # Convert the difference into a 0-100% confidence score
        certainty = (1 - best_score) * 100

        if best_score < cutoff:
            logger.info(f"[INFO] Identity Matched: {matched_person['name']} (Confidence: {certainty:.1f}%)")
            return matched_person, certainty
        
        logger.debug(f"[DEBUG] Face not recognized. Best match was {matched_person['name']} with diff {best_score:.4f}")
        return None, certainty

    def calculate_difference(self, vec1, vec2):
        """ Helper to find the mathematical distance between two faces. """
        # Simple cosine distance formula
        return 1 - np.dot(vec1, vec2)
