from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.settings import CONFIG

"""
Database Session Module
-----------------------
Handles database connection setup and session management.
Provides thread-safe access to the database using SQLAlchemy sessions.
"""

DB_URL = f"sqlite:///{CONFIG['database']['db_path']}"

# Create engine
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

# Synchronous sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency to provide a DB session context.
    Usage:
        with get_db() as session:
            # use session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
