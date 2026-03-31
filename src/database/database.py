import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from src.utils.logger import logger
from config.settings import CONFIG

"""
Database Module
---------------
Provides direct SQLite access for managing users and attendance records.
Uses standard SQL queries for clarity and reliability.
"""

# Fetch DB path from config
DB_PATH = CONFIG.get('database', {}).get('db_path', 'data/attendance.db')

class DatabaseManager:
    """
    Handles all persistent storage operations for the attendance system.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table: Stores identity and facial signature
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        embedding BLOB NOT NULL
                    )
                ''')

                # Attendance table: Logs check-in events
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {str(e)}")

    def add_user(self, name: str, embedding: np.ndarray) -> bool:
        """
        Registers a new user with their facial embedding.
        
        Args:
            name: Person's name.
            embedding: Normalized numpy array of face features.
        """
        try:
            # Serialize numpy array to bytes
            embedding_bytes = embedding.tobytes()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (name, embedding) VALUES (?, ?)",
                    (name, embedding_bytes)
                )
                conn.commit()
                logger.info(f"User '{name}' added successfully.")
                return True
        except sqlite3.Error as e:
            logger.error(f"Error adding user {name}: {str(e)}")
            return False

    def get_all_users(self) -> List[dict]:
        """
        Retrieves all registered users and their embeddings.
        
        Returns:
            List[dict]: [{'id': 1, 'name': 'John', 'embedding': np.array}, ...]
        """
        users = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, embedding FROM users")
                rows = cursor.fetchall()
                
                for row in rows:
                    # Deserialize bytes back to numpy array
                    embedding = np.frombuffer(row[2], dtype=np.float64)
                    users.append({
                        "id": row[0],
                        "name": row[1],
                        "embedding": embedding
                    })
            return users
        except sqlite3.Error as e:
            logger.error(f"Error fetching users: {str(e)}")
            return []

    def mark_attendance(self, user_id: int) -> bool:
        """
        Logs an attendance entry for a user.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO attendance (user_id) VALUES (?)",
                    (user_id,)
                )
                conn.commit()
                logger.info(f"Attendance marked for user_id={user_id}")
                return True
        except sqlite3.Error as e:
            logger.error(f"Error marking attendance for {user_id}: {str(e)}")
            return False

    def check_recent_entry(self, user_id: int, minutes: int = 60) -> bool:
        """
        Avoids redundant entries if the user was already marked recently.
        
        Returns:
            bool: True if they have a record within the specified timeframe.
        """
        try:
            now = datetime.now()
            time_limit = now - timedelta(minutes=minutes)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM attendance WHERE user_id = ? AND timestamp > ? LIMIT 1",
                    (user_id, time_limit.strftime('%Y-%m-%d %H:%M:%S'))
                )
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Error checking recent entry for {user_id}: {str(e)}")
            return False
