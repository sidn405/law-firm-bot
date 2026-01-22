"""
LAWPAY (8AM PAYMENT PLATFORM) INTEGRATION
OAuth-based integration for law firm payment processing
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json
import secrets

class LawPayClient:
    """8am Payment Platform (LawPay) API Client"""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        sandbox: bool = True
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.sandbox = sandbox
        
        # API endpoints
        self.base_url = "https://sandbox.8am.com" if sandbox else "https://api.8am.com"
        self.auth_url = f"{self.base_url}/oauth/authorize"
        self.token_url = f"{self.base_url}/oauth/token"
        
        # Token storage (in production, store in database)
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        
    def get_authorization_url(self, state: str = None) -> str:
        """
        Generate OAuth authorization URL for merchant connection
        
        Args:
            state: Optional state parameter for CSRF protection
            
        Returns:
            Authorization URL to redirect user to
        """
        if not state:
            state = secrets.token_urlsafe(32)
            
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": "payments invoices merchants"  # Adjust scopes as needed
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.auth_url}?{query_string}"
    
    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token
        
        Args:
            code: Authorization code from OAuth callback
            
        Returns:
            Token response containing access_token, refresh_token, etc.
        """
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        response = requests.post(self.token_url, data=payload)
        response.raise_for_status()
        
        token_data = response.json()
        
        # Store tokens
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token")
        
        # Calculate expiration time
        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        return token_data
    
    def refresh_access_token(self) -> Dict[str, Any]:
        """
        Refresh the access token using refresh token
        
        Returns:
            New token response
        """
        if not self.refresh_token:
            raise ValueError("No refresh token available")
            
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        response = requests.post(self.token_url, data=payload)
        response.raise_for_status()
        
        token_data = response.json()
        
        # Update tokens
        self.access_token = token_data.get("access_token")
        if "refresh_token" in token_data:
            self.refresh_token = token_data["refresh_token"]
            
        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        return token_data
    
    def _ensure_valid_token(self):
        """Ensure access token is valid, refresh if needed"""
        if not self.access_token:
            raise ValueError("Not authenticated. Call exchange_code_for_token first.")
            
        # Check if token is about to expire (within 5 minutes)
        if self.token_expires_at and datetime.now() >= (self.token_expires_at - timedelta(minutes=5)):
            self.refresh_access_token()
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make authenticated API request
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: URL parameters
            
        Returns:
            API response as dictionary
        """
        self._ensure_valid_token()
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data,
            params=params
        )
        
        response.raise_for_status()
        return response.json()
    
    # ============================================
    # MERCHANT MANAGEMENT
    # ============================================
    
    def get_connected_merchants(self) -> List[Dict[str, Any]]:
        """
        Get list of connected merchant accounts
        
        Returns:
            List of merchant account details
        """
        return self._make_request("GET", "/api/v1/merchants")
    
    def get_merchant_details(self, merchant_id: str) -> Dict[str, Any]:
        """
        Get details for a specific merchant
        
        Args:
            merchant_id: Merchant account ID
            
        Returns:
            Merchant account details
        """
        return self._make_request("GET", f"/api/v1/merchants/{merchant_id}")
    
    # ============================================
    # PAYMENT FORMS
    # ============================================
    
    def create_payment_form(
        self,
        merchant_id: str,
        amount: float,
        description: str,
        client_name: str,
        client_email: str,
        reference_number: Optional[str] = None,
        custom_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a hosted payment form for client to submit payment
        
        Args:
            merchant_id: Merchant account ID
            amount: Payment amount in dollars
            description: Payment description
            client_name: Client's full name
            client_email: Client's email address
            reference_number: Optional case/matter reference number
            custom_fields: Optional custom field values
            
        Returns:
            Payment form details including URL
        """
        payload = {
            "merchant_id": merchant_id,
            "amount": amount,
            "description": description,
            "client": {
                "name": client_name,
                "email": client_email
            },
            "reference_number": reference_number,
            "custom_fields": custom_fields or {},
            "return_url": f"{os.getenv('BASE_URL', '')}/payment/success",
            "cancel_url": f"{os.getenv('BASE_URL', '')}/payment/cancel"
        }
        
        return self._make_request("POST", "/api/v1/payment-forms", data=payload)
    
    # ============================================
    # INVOICES
    # ============================================
    
    def create_invoice(
        self,
        merchant_id: str,
        client_name: str,
        client_email: str,
        line_items: List[Dict[str, Any]],
        due_date: Optional[str] = None,
        notes: Optional[str] = None,
        send_email: bool = True
    ) -> Dict[str, Any]:
        """
        Create an invoice for client payment
        
        Args:
            merchant_id: Merchant account ID
            client_name: Client's full name
            client_email: Client's email address
            line_items: List of invoice line items
                Example: [{"description": "Consultation Fee", "amount": 500.00, "quantity": 1}]
            due_date: Invoice due date (YYYY-MM-DD)
            notes: Optional invoice notes
            send_email: Whether to send invoice email to client
            
        Returns:
            Invoice details including payment URL
        """
        # Calculate total
        total = sum(item.get("amount", 0) * item.get("quantity", 1) for item in line_items)
        
        payload = {
            "merchant_id": merchant_id,
            "client": {
                "name": client_name,
                "email": client_email
            },
            "line_items": line_items,
            "total": total,
            "due_date": due_date,
            "notes": notes,
            "send_email": send_email
        }
        
        return self._make_request("POST", "/api/v1/invoices", data=payload)
    
    def get_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """
        Get invoice details
        
        Args:
            invoice_id: Invoice ID
            
        Returns:
            Invoice details
        """
        return self._make_request("GET", f"/api/v1/invoices/{invoice_id}")
    
    def get_invoices(
        self,
        merchant_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get list of invoices
        
        Args:
            merchant_id: Filter by merchant ID
            status: Filter by status (pending, paid, overdue, cancelled)
            limit: Maximum number of invoices to return
            
        Returns:
            List of invoices
        """
        params = {
            "limit": limit
        }
        if merchant_id:
            params["merchant_id"] = merchant_id
        if status:
            params["status"] = status
            
        return self._make_request("GET", "/api/v1/invoices", params=params)
    
    def send_invoice_reminder(self, invoice_id: str) -> Dict[str, Any]:
        """
        Send payment reminder for an invoice
        
        Args:
            invoice_id: Invoice ID
            
        Returns:
            Reminder status
        """
        return self._make_request("POST", f"/api/v1/invoices/{invoice_id}/remind")
    
    # ============================================
    # PAYMENTS
    # ============================================
    
    def get_payment(self, payment_id: str) -> Dict[str, Any]:
        """
        Get payment details
        
        Args:
            payment_id: Payment transaction ID
            
        Returns:
            Payment details
        """
        return self._make_request("GET", f"/api/v1/payments/{payment_id}")
    
    def get_payments(
        self,
        merchant_id: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get list of payments
        
        Args:
            merchant_id: Filter by merchant ID
            status: Filter by status (completed, pending, failed, refunded)
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)
            limit: Maximum number of payments to return
            
        Returns:
            List of payments
        """
        params = {
            "limit": limit
        }
        if merchant_id:
            params["merchant_id"] = merchant_id
        if status:
            params["status"] = status
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
            
        return self._make_request("GET", "/api/v1/payments", params=params)
    
    def refund_payment(
        self,
        payment_id: str,
        amount: Optional[float] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refund a payment
        
        Args:
            payment_id: Payment transaction ID
            amount: Refund amount (None for full refund)
            reason: Refund reason
            
        Returns:
            Refund details
        """
        payload = {}
        if amount is not None:
            payload["amount"] = amount
        if reason:
            payload["reason"] = reason
            
        return self._make_request("POST", f"/api/v1/payments/{payment_id}/refund", data=payload)
    
    # ============================================
    # WEBHOOKS
    # ============================================
    
    def create_webhook(
        self,
        url: str,
        events: List[str],
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a webhook subscription
        
        Args:
            url: Webhook endpoint URL
            events: List of events to subscribe to
                Examples: ["payment.completed", "payment.failed", "invoice.paid"]
            description: Optional webhook description
            
        Returns:
            Webhook details including webhook ID and secret
        """
        payload = {
            "url": url,
            "events": events,
            "description": description
        }
        
        return self._make_request("POST", "/api/v1/webhooks", data=payload)
    
    def get_webhooks(self) -> List[Dict[str, Any]]:
        """
        Get list of webhook subscriptions
        
        Returns:
            List of webhooks
        """
        return self._make_request("GET", "/api/v1/webhooks")
    
    def delete_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """
        Delete a webhook subscription
        
        Args:
            webhook_id: Webhook ID
            
        Returns:
            Deletion confirmation
        """
        return self._make_request("DELETE", f"/api/v1/webhooks/{webhook_id}")
    
    @staticmethod
    def verify_webhook_signature(
        payload: str,
        signature: str,
        webhook_secret: str
    ) -> bool:
        """
        Verify webhook signature for security
        
        Args:
            payload: Raw webhook payload
            signature: Signature from X-8am-Signature header
            webhook_secret: Webhook secret key
            
        Returns:
            True if signature is valid
        """
        import hmac
        import hashlib
        
        expected_signature = hmac.new(
            webhook_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_lawpay_client(sandbox: bool = True) -> LawPayClient:
    """
    Create LawPay client instance from environment variables
    
    Args:
        sandbox: Whether to use sandbox environment
        
    Returns:
        Configured LawPayClient instance
    """
    client_id = os.getenv("LAWPAY_CLIENT_ID")
    client_secret = os.getenv("LAWPAY_CLIENT_SECRET")
    redirect_uri = os.getenv("LAWPAY_REDIRECT_URI", f"{os.getenv('BASE_URL')}/api/lawpay/callback")
    
    if not client_id or not client_secret:
        raise ValueError("LAWPAY_CLIENT_ID and LAWPAY_CLIENT_SECRET must be set")
    
    return LawPayClient(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        sandbox=sandbox
    )


def format_invoice_line_items(
    case_type: str,
    consultation_fee: float = 500.00,
    retainer_fee: float = 2500.00
) -> List[Dict[str, Any]]:
    """
    Create standard invoice line items based on case type
    
    Args:
        case_type: Type of legal case
        consultation_fee: Consultation fee amount
        retainer_fee: Retainer fee amount
        
    Returns:
        List of formatted line items
    """
    line_items = [
        {
            "description": f"Initial Consultation - {case_type}",
            "amount": consultation_fee,
            "quantity": 1
        }
    ]
    
    if retainer_fee > 0:
        line_items.append({
            "description": f"Retainer Fee - {case_type}",
            "amount": retainer_fee,
            "quantity": 1
        })
    
    return line_items