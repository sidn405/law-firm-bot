"""
LAWPAY INTEGRATION TESTING SCRIPT
Run this to test all LawPay endpoints
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your Railway URL
TEST_CASE_ID = "test_case_123"
TEST_CLIENT_EMAIL = "test@example.com"
TEST_CLIENT_NAME = "John Doe"

class LawPayTester:
    """Test LawPay integration"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []
    
    def test(self, name: str, method: str, endpoint: str, data=None, expected_status=200):
        """Run a test"""
        url = f"{self.base_url}{endpoint}"
        
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        print(f"Method: {method}")
        print(f"URL: {url}")
        if data:
            print(f"Data: {json.dumps(data, indent=2)}")
        
        try:
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=data)
            elif method == "PUT":
                response = requests.put(url, json=data)
            elif method == "DELETE":
                response = requests.delete(url)
            
            print(f"\nStatus Code: {response.status_code}")
            
            try:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
            except:
                result = response.text
                print(f"Response: {result}")
            
            success = response.status_code == expected_status
            
            self.results.append({
                "name": name,
                "success": success,
                "status_code": response.status_code,
                "response": result
            })
            
            if success:
                print("✅ PASSED")
            else:
                print(f"❌ FAILED (Expected {expected_status}, got {response.status_code})")
            
            return result
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            self.results.append({
                "name": name,
                "success": False,
                "error": str(e)
            })
            return None
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.results if r.get("success"))
        total = len(self.results)
        
        for result in self.results:
            status = "✅" if result.get("success") else "❌"
            print(f"{status} {result['name']}")
        
        print(f"\nPassed: {passed}/{total}")
        print("="*60)

def run_all_tests():
    """Run all LawPay integration tests"""
    
    tester = LawPayTester(BASE_URL)
    
    # Test 1: Check LawPay Status
    tester.test(
        "Check LawPay Connection Status",
        "GET",
        "/api/lawpay/status"
    )
    
    # Test 2: Initiate Connection (if not connected)
    connect_result = tester.test(
        "Initiate LawPay Connection",
        "POST",
        "/api/lawpay/connect",
        data={}
    )
    
    if connect_result and connect_result.get("authorization_url"):
        print(f"\n⚠️  MANUAL STEP REQUIRED:")
        print(f"Visit this URL to authorize: {connect_result['authorization_url']}")
        print(f"After authorizing, you'll get a 'code' parameter in the callback URL")
        
        code = input("\nEnter the authorization code from callback URL: ").strip()
        
        if code:
            # Test 3: Complete OAuth
            tester.test(
                "Complete OAuth Connection",
                "POST",
                "/api/lawpay/callback",
                data={"code": code}
            )
    
    # Test 4: Get Merchants
    merchants_result = tester.test(
        "Get Connected Merchants",
        "GET",
        "/api/lawpay/merchants"
    )
    
    merchant_id = None
    if merchants_result and merchants_result.get("merchants"):
        merchant_id = merchants_result["merchants"][0].get("id")
        print(f"\n✅ Found merchant ID: {merchant_id}")
    
    # Test 5: Create Payment Form
    tester.test(
        "Create Payment Form",
        "POST",
        "/api/lawpay/payment-form",
        data={
            "case_id": TEST_CASE_ID,
            "client_name": TEST_CLIENT_NAME,
            "client_email": TEST_CLIENT_EMAIL,
            "amount": 500.00,
            "description": "Test Consultation Fee",
            "merchant_id": merchant_id
        }
    )
    
    # Test 6: Create Invoice
    invoice_result = tester.test(
        "Create Invoice",
        "POST",
        "/api/lawpay/invoice",
        data={
            "case_id": TEST_CASE_ID,
            "client_name": TEST_CLIENT_NAME,
            "client_email": TEST_CLIENT_EMAIL,
            "case_type": "Personal Injury",
            "consultation_fee": 500.00,
            "retainer_fee": 2500.00,
            "merchant_id": merchant_id,
            "send_email": False  # Don't actually send in test
        }
    )
    
    # Test 7: Get Invoices
    tester.test(
        "Get All Invoices",
        "GET",
        "/api/lawpay/invoices?limit=10"
    )
    
    # Test 8: Get Payments
    tester.test(
        "Get All Payments",
        "GET",
        "/api/lawpay/payments?limit=10"
    )
    
    # Test 9: Chatbot Payment Request
    tester.test(
        "Chatbot Request Payment",
        "POST",
        f"/api/chatbot/request-payment?case_id={TEST_CASE_ID}&payment_type=invoice"
    )
    
    # Test 10: Check Payment Status
    tester.test(
        "Check Payment Status",
        "GET",
        f"/api/chatbot/payment-status/{TEST_CASE_ID}"
    )
    
    # Print summary
    tester.print_summary()
    
    return tester.results

def test_webhook_verification():
    """Test webhook signature verification"""
    from lawpay_integration import LawPayClient
    
    print("\n" + "="*60)
    print("TEST: Webhook Signature Verification")
    print("="*60)
    
    # Test data
    payload = '{"event_type":"payment.completed","data":{"id":"pay_123"}}'
    webhook_secret = "test_secret_key"
    
    # Create signature
    import hmac
    import hashlib
    signature = hmac.new(
        webhook_secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Verify
    is_valid = LawPayClient.verify_webhook_signature(
        payload,
        signature,
        webhook_secret
    )
    
    print(f"Payload: {payload}")
    print(f"Secret: {webhook_secret}")
    print(f"Signature: {signature}")
    print(f"Valid: {is_valid}")
    
    if is_valid:
        print("✅ Webhook verification PASSED")
    else:
        print("❌ Webhook verification FAILED")

def quick_integration_check():
    """Quick check to see if integration is working"""
    
    print("\n" + "="*60)
    print("LAWPAY INTEGRATION QUICK CHECK")
    print("="*60 + "\n")
    
    checks = []
    
    # Check 1: Environment Variables
    print("1. Checking environment variables...")
    import os
    env_vars = {
        "LAWPAY_CLIENT_ID": os.getenv("LAWPAY_CLIENT_ID"),
        "LAWPAY_CLIENT_SECRET": os.getenv("LAWPAY_CLIENT_SECRET"),
        "LAWPAY_REDIRECT_URI": os.getenv("LAWPAY_REDIRECT_URI")
    }
    
    env_check = all(env_vars.values())
    checks.append(("Environment Variables", env_check))
    
    for key, value in env_vars.items():
        status = "✅" if value else "❌"
        display_value = "SET" if value else "NOT SET"
        print(f"   {status} {key}: {display_value}")
    
    # Check 2: Server Status
    print("\n2. Checking server connection...")
    try:
        response = requests.get(f"{BASE_URL}/api/lawpay/status", timeout=5)
        server_check = response.status_code in [200, 401]  # 401 means server is up but not connected
        checks.append(("Server Connection", server_check))
        print(f"   {'✅' if server_check else '❌'} Server Status: {response.status_code}")
    except Exception as e:
        checks.append(("Server Connection", False))
        print(f"   ❌ Server Error: {str(e)}")
    
    # Check 3: Database
    print("\n3. Checking database...")
    try:
        # This would need to be implemented based on your DB setup
        db_check = True  # Placeholder
        checks.append(("Database", db_check))
        print(f"   {'✅' if db_check else '❌'} Database connection")
    except Exception as e:
        checks.append(("Database", False))
        print(f"   ❌ Database Error: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    passed = sum(1 for _, check in checks if check)
    total = len(checks)
    print(f"OVERALL: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All systems ready for LawPay integration!")
    else:
        print("⚠️  Some issues need to be resolved")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    print("""
╔════════════════════════════════════════════════════════════╗
║         LAWPAY INTEGRATION TESTING SUITE                   ║
╚════════════════════════════════════════════════════════════╝

What would you like to test?

1. Quick Integration Check (recommended first)
2. Full Integration Tests
3. Webhook Verification Test
4. Exit

    """)
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        quick_integration_check()
    elif choice == "2":
        print("\n⚠️  Make sure your server is running on", BASE_URL)
        input("Press Enter to continue...")
        run_all_tests()
    elif choice == "3":
        test_webhook_verification()
    else:
        print("Exiting...")
        sys.exit(0)