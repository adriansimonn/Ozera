"""
Database connection and session management.
"""
import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from models.database import Base

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ozera")

# Detect Supabase (remote) vs local Postgres and adjust settings accordingly
_is_remote = "supabase" in DATABASE_URL or DATABASE_URL.startswith("postgresql+psycopg2://") and "@db." in DATABASE_URL

_connect_args: dict = {}
if _is_remote:
    import ssl
    ssl_ctx = ssl.create_default_context()
    _connect_args["ssl_context"] = ssl_ctx

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5 if _is_remote else 10,
    max_overflow=10 if _is_remote else 20,
    echo=False,
    connect_args=_connect_args,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI routes to get database session.

    Usage:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables.
    This should be called on application startup or use Alembic for migrations.
    """
    Base.metadata.create_all(bind=engine)


def drop_db():
    """
    Drop all tables - USE WITH CAUTION!
    Only for development/testing.
    """
    Base.metadata.drop_all(bind=engine)


if __name__ == "__main__":
    # Quick test of database connection
    print(f"Testing database connection to: {DATABASE_URL}")
    try:
        with engine.connect() as conn:
            print("✓ Database connection successful!")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
