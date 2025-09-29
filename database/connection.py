from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import logging

from config import settings
from database.models import Base

logger = logging.getLogger(__name__)

# Database engine
engine = None
SessionLocal = None

async def init_db():
    """Initialize database connection and create tables"""
    global engine, SessionLocal
    
    try:
        logger.info("Initializing database connection...")
        
        # Create database engine
        engine = create_engine(
            settings.DATABASE_URL,
            poolclass=StaticPool,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=settings.DEBUG
        )
        
        # Create session factory
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

async def close_db():
    """Close database connection"""
    global engine
    
    if engine:
        engine.dispose()
        logger.info("Database connection closed")

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def test_connection():
    """Test database connection"""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False