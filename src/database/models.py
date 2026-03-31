from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
import datetime

"""
Database Models Module
----------------------
Defines the database schema using SQLAlchemy ORM.
Includes tables for Users, Sessions, and Attendance records.
"""

Base = declarative_base()

class User(Base):
    """
    Represents a registered user in the attendance system.
    """
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    face_encoding = Column(String, nullable=True) # Serialized encoding

class AttendanceRecord(Base):
    """
    Records an attendance event (check-in/check-out).
    """
    __tablename__ = 'attendance_records'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(20)) # e.g., 'IN' or 'OUT'
