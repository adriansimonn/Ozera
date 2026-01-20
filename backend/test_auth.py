"""
Quick test script to verify authentication is working.
Run this with: python test_auth.py
"""
import sys
sys.path.insert(0, '.')

from sqlalchemy.orm import Session
from db import SessionLocal
from models.database import User, CreditBalance
from services.auth_service import hash_password, verify_password, create_access_token, decode_access_token

def test_database_connection():
    """Test database connection."""
    print("Testing database connection...")
    db = SessionLocal()
    try:
        # Try a simple query
        user_count = db.query(User).count()
        print(f"✓ Database connected successfully! Found {user_count} users.")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False
    finally:
        db.close()

def test_password_hashing():
    """Test password hashing."""
    print("\nTesting password hashing...")
    password = "test_password_123"
    hashed = hash_password(password)

    if verify_password(password, hashed):
        print(f"✓ Password hashing works correctly!")
        return True
    else:
        print(f"✗ Password verification failed!")
        return False

def test_jwt_tokens():
    """Test JWT token generation."""
    print("\nTesting JWT tokens...")
    user_id = 1
    token = create_access_token({"sub": user_id})
    print(f"Generated token: {token[:50]}...")

    decoded = decode_access_token(token)
    if decoded and decoded.get("sub") == user_id:
        print(f"✓ JWT token generation and decoding works!")
        return True
    else:
        print(f"✗ JWT token decoding failed!")
        return False

def test_create_user():
    """Test creating a user."""
    print("\nTesting user creation...")
    db = SessionLocal()
    try:
        # Check if test user already exists
        existing_user = db.query(User).filter(User.email == "test@example.com").first()
        if existing_user:
            print("Test user already exists, deleting...")
            db.delete(existing_user)
            db.commit()

        # Create test user
        user = User(
            email="test@example.com",
            hashed_password=hash_password("testpass123"),
            full_name="Test User",
            is_active=True,
            is_verified=False,
        )
        db.add(user)
        db.flush()

        # Create credit balance
        credit_balance = CreditBalance(
            user_id=user.id,
            balance_usd=0.0,
            reserved_usd=0.0,
        )
        db.add(credit_balance)
        db.commit()

        print(f"✓ Created user with ID {user.id} and email {user.email}")
        print(f"  Credit balance: ${credit_balance.balance_usd}")
        return True

    except Exception as e:
        print(f"✗ User creation failed: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 60)
    print("OZERA AUTHENTICATION SYSTEM TEST")
    print("=" * 60)

    results = []
    results.append(("Database Connection", test_database_connection()))
    results.append(("Password Hashing", test_password_hashing()))
    results.append(("JWT Tokens", test_jwt_tokens()))
    results.append(("User Creation", test_create_user()))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")

    all_passed = all(result[1] for result in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("✓ ALL TESTS PASSED! Phase 1 authentication is working.")
    else:
        print("✗ SOME TESTS FAILED. Please check the errors above.")
    print("=" * 60)
