"""Database session configuration and initialization.
Supports SQLite (local file database) and PostgreSQL.
"""

import os
import bcrypt
if not hasattr(bcrypt, "__about__"):
    bcrypt.__about__ = type("about", (), {"__version__": getattr(bcrypt, "__version__", "4.0.0")})()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database.models import Base, User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./customerpulse.db")

# SQLite connection args for multi-threaded FastAPI
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency for yielding database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def init_db():
    """Create all database tables and seed default admin/analyst users."""
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if admin user exists
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin_user = User(
                username="admin",
                email="admin@customerpulse.ai",
                hashed_password=hash_password("admin123"),
                role="ADMIN",
                is_active=True,
            )
            db.add(admin_user)
            
        analyst = db.query(User).filter(User.username == "analyst").first()
        if not analyst:
            analyst_user = User(
                username="analyst",
                email="analyst@customerpulse.ai",
                hashed_password=hash_password("analyst123"),
                role="ANALYST",
                is_active=True,
            )
            db.add(analyst_user)
            
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Warning during init_db user seeding: {e}")
    finally:
        db.close()
