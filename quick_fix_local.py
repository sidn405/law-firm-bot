#!/usr/bin/env python3
"""
Quick Fix for Payment Verification Issues
Run this once to fix all client emails in the database
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Get database URL
DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL environment variable not set")
    print("Run this script in Railway environment")
    exit(1)

# Fix Railway PostgreSQL URL format
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

print("=" * 60)
print("QUICK FIX FOR CLIENT EMAIL ISSUES")
print("=" * 60)

# Create engine
engine = create_engine(DATABASE_URL)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

try:
    # Get count of clients with problematic emails
    result = db.execute(text("""
        SELECT COUNT(*) as count
        FROM clients
        WHERE email != LOWER(TRIM(email))
    """))
    problem_count = result.fetchone()[0]
    
    print(f"\n📊 Found {problem_count} clients with email issues")
    
    if problem_count > 0:
        print("\n🔧 Fixing emails...")
        
        # Show what will be fixed
        result = db.execute(text("""
            SELECT name, email, LOWER(TRIM(email)) as fixed_email
            FROM clients
            WHERE email != LOWER(TRIM(email))
        """))
        
        print("\nChanges to be made:")
        for row in result:
            print(f"  • {row[0]}: '{row[1]}' → '{row[2]}'")
        
        # Fix the emails
        db.execute(text("""
            UPDATE clients
            SET email = LOWER(TRIM(email))
            WHERE email != LOWER(TRIM(email))
        """))
        
        db.commit()
        print(f"\n✅ Fixed {problem_count} email(s)")
    else:
        print("\n✅ All emails are already correct!")
    
    # Show current state of all clients
    print("\n" + "=" * 60)
    print("CURRENT CLIENTS IN DATABASE")
    print("=" * 60)
    
    result = db.execute(text("""
        SELECT 
            name, 
            email, 
            phone,
            case_type,
            status
        FROM clients
        ORDER BY name
    """))
    
    clients = result.fetchall()
    
    if not clients:
        print("\n⚠️  No clients found in database")
        print("\nCreate test clients by running:")
        print("  python test_clients.py create")
    else:
        print(f"\nTotal clients: {len(clients)}\n")
        for i, client in enumerate(clients, 1):
            print(f"{i}. {client[0]}")
            print(f"   Email: {client[1]}")
            print(f"   Phone: {client[2] or 'N/A'}")
            print(f"   Case: {client[3] or 'N/A'}")
            print(f"   Status: {client[4]}")
            print(f"   Test format: {client[0]}, {client[1]}")
            print()
    
    print("=" * 60)
    print("✅ DATABASE READY FOR TESTING")
    print("=" * 60)
    
    if clients:
        print("\nYou can now test the payment flow with any of the clients above")
        print("Format: Full Name, Email")
        print(f"\nExample: {clients[0][0]}, {clients[0][1]}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    db.rollback()
finally:
    db.close()

print("\n" + "=" * 60)