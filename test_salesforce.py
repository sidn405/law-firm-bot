#!/usr/bin/env python3
"""
Quick test script for Salesforce integration
Run this to verify your credentials before deploying
"""

import os
from simple_salesforce import Salesforce

# Load credentials from environment or replace with your values
SALESFORCE_USERNAME = os.getenv("SALESFORCE_USERNAME", "your-email@example.com")
SALESFORCE_PASSWORD = os.getenv("SALESFORCE_PASSWORD", "your-password")
SALESFORCE_SECURITY_TOKEN = os.getenv("SALESFORCE_SECURITY_TOKEN", "your-token")
SALESFORCE_DOMAIN = os.getenv("SALESFORCE_DOMAIN", "login")

def test_connection():
    """Test Salesforce connection"""
    print("=" * 50)
    print("SALESFORCE CONNECTION TEST")
    print("=" * 50)
    
    print(f"\n📋 Configuration:")
    print(f"   Username: {SALESFORCE_USERNAME}")
    print(f"   Password: {'*' * len(SALESFORCE_PASSWORD)}")
    print(f"   Token: {'*' * len(SALESFORCE_SECURITY_TOKEN)}")
    print(f"   Domain: {SALESFORCE_DOMAIN}")
    
    print(f"\n🔗 Attempting to connect...")
    
    try:
        sf = Salesforce(
            username=SALESFORCE_USERNAME,
            password=SALESFORCE_PASSWORD,
            security_token=SALESFORCE_SECURITY_TOKEN,
            domain=SALESFORCE_DOMAIN
        )
        
        print("✅ Connection successful!")
        
        # Test queries
        print("\n📊 Testing queries...")
        
        # Query accounts
        accounts = sf.query("SELECT Id, Name FROM Account LIMIT 5")
        print(f"   Accounts found: {accounts['totalSize']}")
        if accounts['records']:
            print(f"   Sample: {accounts['records'][0]['Name']}")
        
        # Query leads
        leads = sf.query("SELECT Id, FirstName, LastName, Email FROM Lead LIMIT 5")
        print(f"   Leads found: {leads['totalSize']}")
        if leads['records']:
            lead = leads['records'][0]
            print(f"   Sample: {lead.get('FirstName', '')} {lead.get('LastName', '')}")
        
        # Query contacts
        contacts = sf.query("SELECT Id, FirstName, LastName, Email FROM Contact LIMIT 5")
        print(f"   Contacts found: {contacts['totalSize']}")
        if contacts['records']:
            contact = contacts['records'][0]
            print(f"   Sample: {contact.get('FirstName', '')} {contact.get('LastName', '')}")
        
        print("\n✅ All tests passed! Salesforce is ready to use.")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed!")
        print(f"   Error: {e}")
        print("\n💡 Common issues:")
        print("   1. Username must be your Salesforce login email")
        print("   2. Password + Security Token must be correct")
        print("   3. Security token is NOT your password - get it from email")
        print("   4. Use 'login' for production, 'test' for sandbox")
        print("   5. Ensure API access is enabled for your user")
        return False

def test_create_lead():
    """Test creating a sample lead"""
    print("\n" + "=" * 50)
    print("TEST: CREATE SAMPLE LEAD")
    print("=" * 50)
    
    try:
        sf = Salesforce(
            username=SALESFORCE_USERNAME,
            password=SALESFORCE_PASSWORD,
            security_token=SALESFORCE_SECURITY_TOKEN,
            domain=SALESFORCE_DOMAIN
        )
        
        lead_data = {
            'FirstName': 'Test',
            'LastName': 'Lead',
            'Email': 'testlead@example.com',
            'Phone': '555-123-4567',
            'Company': 'Test Company',
            'LeadSource': 'Website',
            'Status': 'New',
            'Description': 'Test lead created by integration test script'
        }
        
        print(f"\n📝 Creating test lead: {lead_data['FirstName']} {lead_data['LastName']}")
        result = sf.Lead.create(lead_data)
        
        if result.get('success'):
            lead_id = result.get('id')
            print(f"✅ Test lead created successfully!")
            print(f"   Lead ID: {lead_id}")
            print(f"   View in Salesforce: https://{SALESFORCE_DOMAIN}.salesforce.com/{lead_id}")
            
            # Clean up - delete the test lead
            print(f"\n🧹 Cleaning up test lead...")
            sf.Lead.delete(lead_id)
            print(f"✅ Test lead deleted")
            
            return True
        else:
            print(f"❌ Lead creation failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Test connection
    connection_ok = test_connection()
    
    # If connection works, test creating a lead
    if connection_ok:
        input("\nPress Enter to test creating a sample lead (will be deleted after)...")
        test_create_lead()
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)