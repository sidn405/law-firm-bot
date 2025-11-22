"""
Test Client Creation Script for Law Firm Chatbot
Run this in Railway Python console or as a one-time script
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime, timezone

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./law_firm.db")

# Fix Railway PostgreSQL URL format
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# Import models (make sure main.py is in the same directory)
try:
    from main import Client, Base
except ImportError:
    print("Error: Cannot import from main.py. Make sure you're in the correct directory.")
    exit(1)

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_test_clients():
    """Create test clients for payment flow testing"""
    db = SessionLocal()
    
    test_clients = [
        {
            "name": "John Smith",
            "email": "john.smith@test.com",
            "phone": "+1-555-123-4567",
            "case_type": "Personal Injury",
            "status": "active"
        },
        {
            "name": "Jane Doe",
            "email": "jane.doe@test.com",
            "phone": "+1-555-234-5678",
            "case_type": "Family Law",
            "status": "active"
        },
        {
            "name": "Michael Johnson",
            "email": "michael.j@test.com",
            "phone": "+1-555-345-6789",
            "case_type": "Workers Compensation",
            "status": "active"
        },
        {
            "name": "Sarah Williams",
            "email": "sarah.w@test.com",
            "phone": "+1-555-456-7890",
            "case_type": "Business Law",
            "status": "active"
        },
        {
            "name": "Jason Brie",
            "email": "jason.brie@test.com",  # Using test.com for testing
            "phone": "+1-555-567-8901",
            "case_type": "Personal Injury",
            "status": "active"
        }
    ]
    
    created_count = 0
    existing_count = 0
    
    print("=" * 60)
    print("CREATING TEST CLIENTS FOR PAYMENT FLOW TESTING")
    print("=" * 60)
    
    for client_data in test_clients:
        # Check if client already exists
        existing = db.query(Client).filter(Client.email == client_data["email"].lower()).first()
        
        if existing:
            print(f"\n✓ Client already exists: {client_data['name']}")
            print(f"  Email: {existing.email}")
            print(f"  ID: {existing.id}")
            existing_count += 1
        else:
            # Create new client
            new_client = Client(
                name=client_data["name"],
                email=client_data["email"].lower(),  # Ensure lowercase
                phone=client_data["phone"],
                case_type=client_data["case_type"],
                status=client_data["status"]
            )
            
            db.add(new_client)
            db.commit()
            db.refresh(new_client)
            
            print(f"\n✓ Created client: {new_client.name}")
            print(f"  Email: {new_client.email}")
            print(f"  Phone: {new_client.phone}")
            print(f"  ID: {new_client.id}")
            created_count += 1
    
    db.close()
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {created_count} created, {existing_count} already existed")
    print("=" * 60)
    
    return created_count, existing_count


def test_payment_verification():
    """Test the payment verification logic"""
    db = SessionLocal()
    
    print("\n" + "=" * 60)
    print("TESTING PAYMENT VERIFICATION")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "John Smith",
            "email": "john.smith@test.com",
            "first_name": "John",
            "last_name": "Smith"
        },
        {
            "name": "Jane Doe",
            "email": "jane.doe@test.com",
            "first_name": "Jane",
            "last_name": "Doe"
        },
        {
            "name": "Jason Brie",
            "email": "jason.brie@test.com",
            "first_name": "Jason",
            "last_name": "Brie"
        }
    ]
    
    for test in test_cases:
        print(f"\n🔍 Testing: {test['name']} ({test['email']})")
        
        # Simulate the verification logic
        client = db.query(Client).filter(Client.email == test['email'].lower()).first()
        
        if client:
            print(f"  ✓ Client found in database")
            print(f"    DB Name: {client.name}")
            print(f"    DB Email: {client.email}")
            
            # Test name matching
            client_name_parts = client.name.lower().split()
            provided_first = test['first_name'].lower()
            provided_last = test['last_name'].lower()
            
            name_match = (
                provided_first in client_name_parts and
                provided_last in client_name_parts
            )
            
            if name_match:
                print(f"  ✓ Name verification PASSED")
                print(f"    Input format for testing: {test['name']}, {test['email']}")
            else:
                print(f"  ✗ Name verification FAILED")
                print(f"    DB name parts: {client_name_parts}")
                print(f"    Provided first: {provided_first}")
                print(f"    Provided last: {provided_last}")
        else:
            print(f"  ✗ Client NOT found in database")
    
    db.close()
    
    print("\n" + "=" * 60)


def list_all_clients():
    """List all clients in the database"""
    db = SessionLocal()
    
    clients = db.query(Client).all()
    
    print("\n" + "=" * 60)
    print(f"ALL CLIENTS IN DATABASE ({len(clients)} total)")
    print("=" * 60)
    
    if not clients:
        print("\nNo clients found in database.")
    else:
        for i, client in enumerate(clients, 1):
            print(f"\n{i}. {client.name}")
            print(f"   ID: {client.id}")
            print(f"   Email: {client.email}")
            print(f"   Phone: {client.phone or 'N/A'}")
            print(f"   Case Type: {client.case_type or 'N/A'}")
            print(f"   Status: {client.status}")
            print(f"   Created: {client.created_at}")
            print(f"   Test format: {client.name}, {client.email}")
    
    db.close()
    
    print("\n" + "=" * 60)


def fix_existing_client_email(old_email, new_email):
    """Fix email for existing client (lowercase it properly)"""
    db = SessionLocal()
    
    client = db.query(Client).filter(Client.email == old_email).first()
    
    if client:
        print(f"\n✓ Found client: {client.name}")
        print(f"  Old email: {client.email}")
        client.email = new_email.lower()
        db.commit()
        print(f"  New email: {client.email}")
        print("  ✓ Email updated successfully")
    else:
        print(f"\n✗ Client not found with email: {old_email}")
    
    db.close()


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("LAW FIRM CHATBOT - TEST CLIENT MANAGER")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "create":
            create_test_clients()
            test_payment_verification()
        elif command == "list":
            list_all_clients()
        elif command == "test":
            test_payment_verification()
        elif command == "fix":
            if len(sys.argv) == 4:
                fix_existing_client_email(sys.argv[2], sys.argv[3])
            else:
                print("Usage: python test_clients.py fix OLD_EMAIL NEW_EMAIL")
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  create - Create test clients and verify")
            print("  list   - List all clients")
            print("  test   - Test payment verification")
            print("  fix    - Fix email for existing client")
    else:
        # Default: create and test
        create_test_clients()
        test_payment_verification()
        list_all_clients()
        
        print("\n" + "=" * 60)
        print("TEST CLIENTS READY!")
        print("=" * 60)
        print("\nYou can now test the payment flow with:")
        print("  • John Smith, john.smith@test.com")
        print("  • Jane Doe, jane.doe@test.com")
        print("  • Jason Brie, jason.brie@test.com")
        print("\n" + "=" * 60)