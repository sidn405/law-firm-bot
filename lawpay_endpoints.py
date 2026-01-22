"""
LAWPAY FASTAPI ENDPOINTS
Add these endpoints to your main.py file
"""

from fastapi import HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import json

# Import the LawPay client
from lawpay_integration import LawPayClient, create_lawpay_client, format_invoice_line_items

# ============================================
# PYDANTIC MODELS
# ============================================

class LawPayConnectionRequest(BaseModel):
    """Request to initiate LawPay OAuth connection"""
    state: Optional[str] = None

class LawPayCallbackRequest(BaseModel):
    """OAuth callback data"""
    code: str
    state: Optional[str] = None

class LawPayPaymentFormRequest(BaseModel):
    """Request to create payment form"""
    case_id: str
    client_name: str
    client_email: str
    amount: float
    description: str
    merchant_id: Optional[str] = None
    reference_number: Optional[str] = None

class LawPayInvoiceRequest(BaseModel):
    """Request to create invoice"""
    case_id: str
    client_name: str
    client_email: str
    case_type: str
    consultation_fee: float = 500.00
    retainer_fee: float = 2500.00
    merchant_id: Optional[str] = None
    due_days: int = 30
    notes: Optional[str] = None
    send_email: bool = True

class LawPayRefundRequest(BaseModel):
    """Request to refund payment"""
    payment_id: str
    amount: Optional[float] = None
    reason: Optional[str] = None

# ============================================
# DATABASE MODEL FOR LAWPAY TOKENS
# ============================================

from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

# Add this to your existing database models in main.py

class LawPayToken(Base):
    """Store LawPay OAuth tokens"""
    __tablename__ = "lawpay_tokens"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text)
    token_type = Column(String, default="Bearer")
    expires_at = Column(DateTime, nullable=False)
    scope = Column(String)
    merchant_id = Column(String)  # Primary merchant ID
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_lawpay_client_from_db(db: Session) -> LawPayClient:
    """
    Get authenticated LawPay client from database tokens
    
    Args:
        db: Database session
        
    Returns:
        Configured LawPayClient with valid tokens
    """
    # Get the most recent token
    token_record = db.query(LawPayToken).order_by(LawPayToken.created_at.desc()).first()
    
    if not token_record:
        raise HTTPException(
            status_code=401,
            detail="LawPay not connected. Please connect your LawPay account first."
        )
    
    # Create client
    client = create_lawpay_client(sandbox=True)  # Set to False for production
    
    # Set tokens
    client.access_token = token_record.access_token
    client.refresh_token = token_record.refresh_token
    client.token_expires_at = token_record.expires_at
    
    # Check if token needs refresh
    if datetime.now() >= (token_record.expires_at - timedelta(minutes=5)):
        # Refresh token
        new_token_data = client.refresh_access_token()
        
        # Update database
        token_record.access_token = new_token_data["access_token"]
        if "refresh_token" in new_token_data:
            token_record.refresh_token = new_token_data["refresh_token"]
        token_record.expires_at = datetime.now() + timedelta(seconds=new_token_data.get("expires_in", 3600))
        token_record.updated_at = datetime.now(timezone.utc)
        db.commit()
    
    return client

def save_lawpay_tokens(db: Session, token_data: dict, merchant_id: Optional[str] = None):
    """
    Save LawPay OAuth tokens to database
    
    Args:
        db: Database session
        token_data: Token response from OAuth
        merchant_id: Optional primary merchant ID
    """
    expires_in = token_data.get("expires_in", 3600)
    expires_at = datetime.now() + timedelta(seconds=expires_in)
    
    token_record = LawPayToken(
        access_token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        token_type=token_data.get("token_type", "Bearer"),
        expires_at=expires_at,
        scope=token_data.get("scope"),
        merchant_id=merchant_id
    )
    
    db.add(token_record)
    db.commit()
    db.refresh(token_record)
    
    return token_record

# ============================================
# FASTAPI ENDPOINTS
# ============================================

@app.post("/api/lawpay/connect")
async def lawpay_connect(request: LawPayConnectionRequest):
    """
    Initiate LawPay OAuth connection
    
    Returns authorization URL to redirect user to
    """
    try:
        client = create_lawpay_client(sandbox=True)  # Set to False for production
        auth_url = client.get_authorization_url(state=request.state)
        
        return {
            "success": True,
            "authorization_url": auth_url,
            "message": "Redirect user to this URL to authorize LawPay access"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate LawPay connection: {str(e)}")

@app.post("/api/lawpay/callback")
async def lawpay_callback(request: LawPayCallbackRequest, db: Session = Depends(get_db)):
    """
    Handle OAuth callback from LawPay
    
    Exchange authorization code for access token
    """
    try:
        client = create_lawpay_client(sandbox=True)
        
        # Exchange code for token
        token_data = client.exchange_code_for_token(request.code)
        
        # Get merchant info
        merchants = client.get_connected_merchants()
        primary_merchant_id = merchants[0]["id"] if merchants else None
        
        # Save tokens to database
        save_lawpay_tokens(db, token_data, primary_merchant_id)
        
        return {
            "success": True,
            "message": "LawPay connected successfully!",
            "merchants": merchants,
            "primary_merchant_id": primary_merchant_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to complete LawPay connection: {str(e)}")

@app.get("/api/lawpay/status")
async def lawpay_status(db: Session = Depends(get_db)):
    """
    Check LawPay connection status
    """
    try:
        token_record = db.query(LawPayToken).order_by(LawPayToken.created_at.desc()).first()
        
        if not token_record:
            return {
                "connected": False,
                "message": "LawPay not connected"
            }
        
        is_expired = datetime.now() >= token_record.expires_at
        
        return {
            "connected": True,
            "expires_at": token_record.expires_at.isoformat(),
            "is_expired": is_expired,
            "merchant_id": token_record.merchant_id,
            "message": "LawPay connected and active" if not is_expired else "Token expired, will auto-refresh on next use"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check LawPay status: {str(e)}")

@app.get("/api/lawpay/merchants")
async def get_merchants(db: Session = Depends(get_db)):
    """
    Get list of connected merchant accounts
    """
    try:
        client = get_lawpay_client_from_db(db)
        merchants = client.get_connected_merchants()
        
        return {
            "success": True,
            "merchants": merchants,
            "count": len(merchants)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get merchants: {str(e)}")

@app.post("/api/lawpay/payment-form")
async def create_payment_form(
    request: LawPayPaymentFormRequest,
    db: Session = Depends(get_db)
):
    """
    Create a hosted payment form for client
    
    Returns payment form URL that can be sent to client
    """
    try:
        client = get_lawpay_client_from_db(db)
        
        # Get merchant ID
        merchant_id = request.merchant_id
        if not merchant_id:
            token_record = db.query(LawPayToken).order_by(LawPayToken.created_at.desc()).first()
            merchant_id = token_record.merchant_id
            
        if not merchant_id:
            raise HTTPException(status_code=400, detail="No merchant ID available")
        
        # Create payment form
        payment_form = client.create_payment_form(
            merchant_id=merchant_id,
            amount=request.amount,
            description=request.description,
            client_name=request.client_name,
            client_email=request.client_email,
            reference_number=request.reference_number or request.case_id
        )
        
        # Update payment record in database
        payment = Payment(
            case_id=request.case_id,
            amount=request.amount,
            status="pending",
            provider="lawpay",
            transaction_id=payment_form.get("id"),
            payment_metadata={
                "form_url": payment_form.get("url"),
                "merchant_id": merchant_id,
                "client_name": request.client_name,
                "client_email": request.client_email
            }
        )
        db.add(payment)
        db.commit()
        
        return {
            "success": True,
            "payment_form": payment_form,
            "payment_url": payment_form.get("url"),
            "payment_id": payment.id,
            "message": "Payment form created. Send the payment_url to your client."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create payment form: {str(e)}")

@app.post("/api/lawpay/invoice")
async def create_invoice(
    request: LawPayInvoiceRequest,
    db: Session = Depends(get_db)
):
    """
    Create an invoice for client payment
    
    Invoice will be emailed to client automatically
    """
    try:
        client = get_lawpay_client_from_db(db)
        
        # Get merchant ID
        merchant_id = request.merchant_id
        if not merchant_id:
            token_record = db.query(LawPayToken).order_by(LawPayToken.created_at.desc()).first()
            merchant_id = token_record.merchant_id
            
        if not merchant_id:
            raise HTTPException(status_code=400, detail="No merchant ID available")
        
        # Format line items
        line_items = format_invoice_line_items(
            case_type=request.case_type,
            consultation_fee=request.consultation_fee,
            retainer_fee=request.retainer_fee
        )
        
        # Calculate due date
        due_date = (datetime.now() + timedelta(days=request.due_days)).strftime("%Y-%m-%d")
        
        # Create invoice
        invoice = client.create_invoice(
            merchant_id=merchant_id,
            client_name=request.client_name,
            client_email=request.client_email,
            line_items=line_items,
            due_date=due_date,
            notes=request.notes,
            send_email=request.send_email
        )
        
        # Calculate total
        total_amount = sum(item["amount"] * item.get("quantity", 1) for item in line_items)
        
        # Update payment record in database
        payment = Payment(
            case_id=request.case_id,
            amount=total_amount,
            status="pending",
            provider="lawpay",
            transaction_id=invoice.get("id"),
            payment_metadata={
                "invoice_url": invoice.get("url"),
                "merchant_id": merchant_id,
                "client_name": request.client_name,
                "client_email": request.client_email,
                "line_items": line_items,
                "due_date": due_date
            }
        )
        db.add(payment)
        db.commit()
        
        return {
            "success": True,
            "invoice": invoice,
            "invoice_url": invoice.get("url"),
            "payment_id": payment.id,
            "total_amount": total_amount,
            "message": f"Invoice created and {'sent to client' if request.send_email else 'ready to send'}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create invoice: {str(e)}")

@app.get("/api/lawpay/invoices")
async def get_invoices(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get list of invoices
    
    Args:
        status: Filter by status (pending, paid, overdue, cancelled)
        limit: Maximum number of invoices to return
    """
    try:
        client = get_lawpay_client_from_db(db)
        invoices = client.get_invoices(status=status, limit=limit)
        
        return {
            "success": True,
            "invoices": invoices,
            "count": len(invoices)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get invoices: {str(e)}")

@app.get("/api/lawpay/payments")
async def get_payments(
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get list of payments
    
    Args:
        status: Filter by status (completed, pending, failed, refunded)
        start_date: Filter by start date (YYYY-MM-DD)
        end_date: Filter by end date (YYYY-MM-DD)
        limit: Maximum number of payments to return
    """
    try:
        client = get_lawpay_client_from_db(db)
        payments = client.get_payments(
            status=status,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        return {
            "success": True,
            "payments": payments,
            "count": len(payments)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get payments: {str(e)}")

@app.post("/api/lawpay/refund")
async def refund_payment(
    request: LawPayRefundRequest,
    db: Session = Depends(get_db)
):
    """
    Refund a payment
    """
    try:
        client = get_lawpay_client_from_db(db)
        
        # Process refund
        refund = client.refund_payment(
            payment_id=request.payment_id,
            amount=request.amount,
            reason=request.reason
        )
        
        # Update payment record
        payment = db.query(Payment).filter(Payment.transaction_id == request.payment_id).first()
        if payment:
            payment.status = "refunded"
            metadata = payment.payment_metadata or {}
            metadata["refund"] = {
                "refund_id": refund.get("id"),
                "amount": request.amount or refund.get("amount"),
                "reason": request.reason,
                "refunded_at": datetime.now().isoformat()
            }
            payment.payment_metadata = metadata
            db.commit()
        
        return {
            "success": True,
            "refund": refund,
            "message": "Payment refunded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refund payment: {str(e)}")

@app.post("/api/lawpay/webhook")
async def lawpay_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Handle LawPay webhook events
    
    This endpoint should be registered in LawPay dashboard
    """
    try:
        # Get webhook signature for verification
        signature = request.headers.get("X-8am-Signature")
        webhook_secret = os.getenv("LAWPAY_WEBHOOK_SECRET", "")
        
        # Get raw body
        body = await request.body()
        payload = body.decode()
        
        # Verify signature
        if webhook_secret and signature:
            if not LawPayClient.verify_webhook_signature(payload, signature, webhook_secret):
                raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse webhook data
        event_data = json.loads(payload)
        event_type = event_data.get("event_type")
        event_object = event_data.get("data", {})
        
        # Handle different event types
        if event_type == "payment.completed":
            # Update payment status
            transaction_id = event_object.get("id")
            payment = db.query(Payment).filter(Payment.transaction_id == transaction_id).first()
            
            if payment:
                payment.status = "completed"
                metadata = payment.payment_metadata or {}
                metadata["completed_at"] = datetime.now().isoformat()
                metadata["webhook_event"] = event_data
                payment.payment_metadata = metadata
                db.commit()
                
                # Update case status
                case = db.query(Case).filter(Case.id == payment.case_id).first()
                if case:
                    case.status = "payment_received"
                    db.commit()
        
        elif event_type == "payment.failed":
            # Update payment status
            transaction_id = event_object.get("id")
            payment = db.query(Payment).filter(Payment.transaction_id == transaction_id).first()
            
            if payment:
                payment.status = "failed"
                metadata = payment.payment_metadata or {}
                metadata["failed_at"] = datetime.now().isoformat()
                metadata["failure_reason"] = event_object.get("failure_reason")
                metadata["webhook_event"] = event_data
                payment.payment_metadata = metadata
                db.commit()
        
        elif event_type == "invoice.paid":
            # Update invoice payment status
            invoice_id = event_object.get("id")
            payment = db.query(Payment).filter(Payment.transaction_id == invoice_id).first()
            
            if payment:
                payment.status = "completed"
                metadata = payment.payment_metadata or {}
                metadata["paid_at"] = datetime.now().isoformat()
                metadata["webhook_event"] = event_data
                payment.payment_metadata = metadata
                db.commit()
        
        return {
            "success": True,
            "event_type": event_type,
            "message": "Webhook processed successfully"
        }
        
    except Exception as e:
        # Log error but return 200 to prevent webhook retries
        print(f"Webhook error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# CHATBOT INTEGRATION
# ============================================

@app.post("/api/chatbot/request-payment")
async def chatbot_request_payment(
    case_id: str,
    payment_type: str = "invoice",  # "invoice" or "payment_form"
    db: Session = Depends(get_db)
):
    """
    Create payment request from chatbot conversation
    
    This endpoint can be called by the chatbot when it determines
    the client is ready to make a payment
    """
    try:
        # Get case details
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get client details
        client = db.query(Client).filter(Client.id == case.client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        client_obj = get_lawpay_client_from_db(db)
        
        # Get merchant ID
        token_record = db.query(LawPayToken).order_by(LawPayToken.created_at.desc()).first()
        merchant_id = token_record.merchant_id
        
        if payment_type == "invoice":
            # Create invoice
            line_items = format_invoice_line_items(
                case_type=case.case_type,
                consultation_fee=500.00,
                retainer_fee=2500.00
            )
            
            due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            
            invoice = client_obj.create_invoice(
                merchant_id=merchant_id,
                client_name=client.name,
                client_email=client.email,
                line_items=line_items,
                due_date=due_date,
                notes=f"Invoice for {case.case_type} case",
                send_email=True
            )
            
            total_amount = sum(item["amount"] * item.get("quantity", 1) for item in line_items)
            
            payment = Payment(
                case_id=case_id,
                amount=total_amount,
                status="pending",
                provider="lawpay",
                transaction_id=invoice.get("id"),
                payment_metadata={
                    "type": "invoice",
                    "invoice_url": invoice.get("url"),
                    "line_items": line_items
                }
            )
            db.add(payment)
            db.commit()
            
            return {
                "success": True,
                "payment_type": "invoice",
                "invoice_url": invoice.get("url"),
                "amount": total_amount,
                "message": f"Invoice sent to {client.email}"
            }
        
        else:  # payment_form
            # Create payment form
            payment_form = client_obj.create_payment_form(
                merchant_id=merchant_id,
                amount=500.00,  # Default consultation fee
                description=f"Consultation Fee - {case.case_type}",
                client_name=client.name,
                client_email=client.email,
                reference_number=case_id
            )
            
            payment = Payment(
                case_id=case_id,
                amount=500.00,
                status="pending",
                provider="lawpay",
                transaction_id=payment_form.get("id"),
                payment_metadata={
                    "type": "payment_form",
                    "form_url": payment_form.get("url")
                }
            )
            db.add(payment)
            db.commit()
            
            return {
                "success": True,
                "payment_type": "payment_form",
                "payment_url": payment_form.get("url"),
                "amount": 500.00,
                "message": f"Payment link created for {client.email}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create payment request: {str(e)}")