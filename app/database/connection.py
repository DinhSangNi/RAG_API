"""
Database Connection Configuration
SQLAlchemy engine and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=settings.DEBUG,
    pool_size=2,           # Minimum number of connections
    max_overflow=28        # Maximum overflow connections (total max = 30)
)

# Create SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for SQLAlchemy models
Base = declarative_base()


def get_db():
    """
    FastAPI dependency to get database session

    Yields:
        Database session that gets closed automatically
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
