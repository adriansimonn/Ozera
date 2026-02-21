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

# Get database connection config from environment.
# Supabase pooler usernames contain dots (e.g. postgres.projectid) which
# break SQLAlchemy's URL parser, so we support separate env vars.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ozera")
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_HOST = os.getenv("DATABASE_HOST")

_is_remote = DATABASE_HOST and "supabase" in DATABASE_HOST or "supabase" in DATABASE_URL

_connect_args: dict = {}
if _is_remote:
    _connect_args["sslmode"] = "require"

# If separate DB env vars are set, use them to bypass URL parsing issues
if DATABASE_USER and DATABASE_HOST:
    _connect_args.update({
        "user": DATABASE_USER,
        "password": os.getenv("DATABASE_PASSWORD", ""),
        "host": DATABASE_HOST,
        "port": os.getenv("DATABASE_PORT", "5432"),
        "dbname": os.getenv("DATABASE_NAME", "postgres"),
    })
    _engine_url = "postgresql+psycopg2://"
else:
    _engine_url = DATABASE_URL

# Create engine
engine = create_engine(
    _engine_url,
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
            print("Database connection successful!")
    except Exception as e:
        print(f"Database connection failed: {e}")
