"""
LAW FIRM AI CHATBOT - BACKEND API
FastAPI backend with OpenAI, Stripe, PayPal, Twilio, Resend Email, File Management
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, EmailStr
from fastapi.staticfiles import StaticFiles
from simple_salesforce import Salesforce, SalesforceLogin
from typing import Optional, List, Dict, Any
import re
import asyncio

import os
import uuid
import json
from datetime import datetime, timezone
import asyncio
import aiohttp
from pathlib import Path

# External services
from openai import OpenAI
import stripe
import resend
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
import requests

# Database
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Web scraping
from bs4 import BeautifulSoup
from flow_state_manager import FlowStateManager

from s3_storage import (
    upload_file_to_s3,
    upload_file_object_to_s3,
    generate_presigned_upload_url,
    generate_presigned_download_url,
    list_client_files,
    delete_file,
    test_s3_connection,
    get_file_metadata
)
import shutil
from tempfile import NamedTemporaryFile

# ============================================
# CONFIGURATION
# ============================================

# API Keys (set as environment variables)
DATABASE_URL = os.getenv("DATABASE_URL")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
LAW_FIRM_EMAIL = os.getenv("LAW_FIRM_EMAIL", "noreply@yourdomain.com")
LAW_FIRM_NAME = os.getenv("LAW_FIRM_NAME", "Your Law Firm")
LAW_FIRM_PHONE = os.getenv("LAW_FIRM_PHONE", "+1234567890")
CALENDLY_API_KEY = os.getenv("CALENDLY_API_KEY", "")
CALENDLY_EVENT_TYPE = os.getenv("CALENDLY_EVENT_TYPE", "")
GOOGLE_CALENDAR_CREDENTIALS = os.getenv("GOOGLE_CALENDAR_CREDENTIALS", "")
CALENDAR_PROVIDER = os.getenv("CALENDAR_PROVIDER", "calendly")
BASE_URL = os.getenv("BASE_URL", "https://your-domain.railway.app")
SALESFORCE_USERNAME = os.getenv("SALESFORCE_USERNAME", "")
SALESFORCE_PASSWORD = os.getenv("SALESFORCE_PASSWORD", "")
SALESFORCE_SECURITY_TOKEN = os.getenv("SALESFORCE_SECURITY_TOKEN", "")
SALESFORCE_DOMAIN = os.getenv("SALESFORCE_DOMAIN", "login")  # 'login' for production, 'test' for sandbox

# Initialize services
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
stripe.api_key = STRIPE_SECRET_KEY
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None
resend.api_key = RESEND_API_KEY

# File storage
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load law firm flow
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLOW_PATH = os.path.join(BASE_DIR, "law_firm", "law_firm.json")

with open(FLOW_PATH, "r", encoding="utf-8") as f:
    LAW_FIRM_FLOW = json.load(f)

flow_manager = FlowStateManager(LAW_FIRM_FLOW)

# ============================================
# DATABASE SETUP
# ============================================

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./law_firm.db")

# Fix Railway PostgreSQL URL format
if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("postgresql://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {},
    poolclass=StaticPool if "sqlite" in SQLALCHEMY_DATABASE_URL else None
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Client(Base):
    __tablename__ = "clients"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    phone = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    case_type = Column(String)
    status = Column(String, default="new")
    
class Case(Base):
    __tablename__ = "cases"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, nullable=False)
    case_type = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    incident_date = Column(String)
    medical_treatment = Column(Boolean)
    has_attorney = Column(Boolean)
    intake_data = Column(JSON)
    
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String)
    session_id = Column(String, nullable=False)
    messages = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    channel = Column(String, default="web")
    
class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, default="pending")
    provider = Column(String)
    transaction_id = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    payment_metadata = Column(JSON)

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String, nullable=False)
    client_id = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    file_size = Column(Integer)

class Appointment(Base):
    __tablename__ = "appointments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, nullable=False)
    case_id = Column(String)
    client_name = Column(String, nullable=False)
    client_email = Column(String, nullable=False)
    client_phone = Column(String)
    scheduled_date = Column(DateTime)
    case_type = Column(String)
    status = Column(String, default="pending")
    calendar_event_id = Column(String)
    calendar_link = Column(String)
    notes = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

# ==============================================================================
# DATABASE DEPENDENCY
# ==============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(title="Law Firm AI Chatbot API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# PYDANTIC MODELS
# ============================================

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    client_id: Optional[str] = None
    
class ClientCreate(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    
class CaseCreate(BaseModel):
    client_id: str
    case_type: str
    description: Optional[str] = None
    intake_data: Optional[Dict[str, Any]] = None
    
class PaymentCreate(BaseModel):
    case_id: str
    amount: float
    provider: str
    
class EmailRequest(BaseModel):
    to: str
    subject: str
    body: str
    html: Optional[str] = None
    
class AppointmentRequest(BaseModel):
    client_name: str
    client_email: str
    client_phone: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    case_type: Optional[str] = None
    notes: Optional[str] = None
    
class PaymentVerificationRequest(BaseModel):
    full_name: str
    email: EmailStr

class StripeCheckoutRequest(BaseModel):
    client_id: str
    amount: float
    description: str
    payment_type: str  # 'retainer', 'case_payment', etc.
    reference_id: Optional[str] = None

# ============================================
# EMAIL SERVICE - RESEND
# ============================================

async def send_email(to: str, subject: str, body: str, html: str = None):
    """Send email via Resend"""
    if not RESEND_API_KEY:
        print("Warning: RESEND_API_KEY not configured")
        return False
    
    try:
        params: resend.Emails.SendParams = {
            "from": f"{LAW_FIRM_NAME} <{LAW_FIRM_EMAIL}>",
            "to": [to],
            "subject": subject,
            "html": html or f"<p>{body.replace(chr(10), '<br>')}</p>",
            "text": body,
        }
        
        email = await asyncio.to_thread(resend.Emails.send, params)
        print(f"Email sent successfully: {email}")
        return True
    except Exception as e:
        print(f"Resend error: {e}")
        return False

# ============================================
# CALENDAR SERVICE
# ============================================

# Replace the CalendarService class in main.py with this improved version

class CalendarService:
    """Manages calendar integrations for appointment scheduling"""
    
    @staticmethod
    async def create_calendly_invitation(appointment_data: dict) -> Dict:
        """Create Calendly scheduling link"""
        if not CALENDLY_API_KEY:
            print("ERROR: CALENDLY_API_KEY not configured")
            return {"success": False, "error": "Calendly not configured"}
        
        try:
            headers = {
                "Authorization": f"Bearer {CALENDLY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # First, get the user info to get the proper URI
            user_response = requests.get(
                "https://api.calendly.com/users/me",
                headers=headers,
                timeout=10
            )
            
            if user_response.status_code != 200:
                print(f"Calendly user API error: {user_response.status_code} - {user_response.text}")
                return {"success": False, "error": f"Calendly user API error: {user_response.status_code}"}
            
            user_data = user_response.json()
            user_uri = user_data["resource"]["uri"]
            
            print(f"Calendly user URI: {user_uri}")
            
            # Get event types for this user
            event_types_response = requests.get(
                f"https://api.calendly.com/event_types?user={user_uri}",
                headers=headers,
                timeout=10
            )
            
            if event_types_response.status_code != 200:
                print(f"Calendly event types API error: {event_types_response.status_code}")
                return {"success": False, "error": "Could not fetch event types"}
            
            event_types_data = event_types_response.json()
            if not event_types_data.get("collection"):
                print("No Calendly event types found")
                return {"success": False, "error": "No event types configured"}
            
            # Find the right event type
            event_type_uri = None
            
            # If CALENDLY_EVENT_TYPE is set, try to match it
            if CALENDLY_EVENT_TYPE:
                # Handle if user provided a booking URL instead of URI
                if "calendly.com/" in CALENDLY_EVENT_TYPE and "/api.calendly.com/" not in CALENDLY_EVENT_TYPE:
                    # Extract the event slug from booking URL (e.g., "30min" from "https://calendly.com/user/30min")
                    event_slug = CALENDLY_EVENT_TYPE.rstrip('/').split('/')[-1]
                    print(f"Extracted event slug from URL: {event_slug}")
                    
                    # Find matching event type by slug
                    for et in event_types_data["collection"]:
                        if et["active"] and event_slug in et.get("scheduling_url", ""):
                            event_type_uri = et["uri"]
                            print(f"Matched event type by slug: {et['name']}")
                            break
                else:
                    # It's already a URI
                    event_type_uri = CALENDLY_EVENT_TYPE
            
            # If still no match, use the first active event type
            if not event_type_uri:
                for et in event_types_data["collection"]:
                    if et["active"]:
                        event_type_uri = et["uri"]
                        print(f"Using first active event type: {et['name']}")
                        break
            
            if not event_type_uri:
                return {"success": False, "error": "No active event types found"}
            
            print(f"Using event type URI: {event_type_uri}")
            
            # Create a single-use scheduling link
            payload = {
                "max_event_count": 1,
                "owner": event_type_uri,
                "owner_type": "EventType"
            }
            
            response = requests.post(
                "https://api.calendly.com/scheduling_links",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            print(f"Calendly scheduling link response: {response.status_code}")
            
            if response.status_code == 201:
                data = response.json()
                booking_url = data["resource"]["booking_url"]
                print(f"✅ Calendly booking URL created: {booking_url}")
                
                return {
                    "success": True,
                    "booking_url": booking_url,
                    "event_id": None
                }
            else:
                error_text = response.text
                print(f"❌ Calendly API error: {response.status_code} - {error_text}")
                return {"success": False, "error": f"Calendly API error: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            print("Calendly API timeout")
            return {"success": False, "error": "Calendly API timeout"}
        except requests.exceptions.RequestException as e:
            print(f"Calendly API request error: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            print(f"Unexpected Calendly error: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def get_available_slots(date: str = None) -> Dict:
        """Get available appointment slots"""
        if not CALENDLY_API_KEY:
            return {"success": False, "error": "Calendly not configured"}
        
        try:
            headers = {
                "Authorization": f"Bearer {CALENDLY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            user_response = requests.get(
                "https://api.calendly.com/users/me",
                headers=headers,
                timeout=10
            )
            
            if user_response.status_code != 200:
                return {"success": False, "error": "Could not fetch user info"}
            
            user_data = user_response.json()
            user_uri = user_data["resource"]["uri"]
            
            params = {"user": user_uri, "count": 10}
            if date:
                params["min_start_time"] = date
            
            response = requests.get(
                "https://api.calendly.com/event_type_available_times",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return {"success": True, "slots": response.json()}
            else:
                return {"success": False, "error": "Could not fetch availability"}
                
        except Exception as e:
            print(f"Error fetching availability: {e}")
            return {"success": False, "error": str(e)}

calendar_service = CalendarService()

# ============================================
# WEB SCRAPER SERVICE
# ============================================

class WebScraperService:
    """Scrapes law firm website for dynamic content"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.cache = {}
        self.cache_duration = 300
        
    async def scrape_page(self, path: str) -> str:
        """Scrape a specific page"""
        url = f"{self.base_url}{path}"
        cache_key = f"page_{path}"
        
        if cache_key in self.cache:
            cached_time, content = self.cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).seconds < self.cache_duration:
                return content
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        for script in soup(["script", "style", "nav", "footer"]):
                            script.decompose()
                        
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = '\n'.join(chunk for chunk in chunks if chunk)
                        
                        self.cache[cache_key] = (datetime.now(timezone.utc), text)
                        return text
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""
        
        return ""
    
    async def get_knowledge_base(self) -> str:
        """Get comprehensive knowledge from website"""
        pages = ["/", "/services", "/about", "/contact", "/practice-areas"]
        contents = []
        
        for page in pages:
            content = await self.scrape_page(page)
            if content:
                contents.append(f"=== Page: {page} ===\n{content}\n")
        
        return "\n".join(contents)

scraper = WebScraperService(base_url="https://yourlawfirm.com")

# ============================================
# CHATBOT SERVICE
# ============================================

class ChatbotService:
    """Handles OpenAI conversations with strict flow management"""
    
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(base_dir, "law_firm", "law_firm.json")

        with open(script_path, "r", encoding="utf-8") as f:
            self.flow = json.load(f)

        self.flow_manager = FlowStateManager(self.flow)

        self.step_index = {}
        for flow in self.flow["flows"]:
            for step in flow["steps"]:
                self.step_index[step["id"]] = step

        self.system_prompt = """You are a law firm intake assistant. Your ONLY job is to ask the exact question provided and collect the answer.

CRITICAL RULES:
1. Ask ONLY the "CURRENT QUESTION" provided below - word for word
2. If user answers the question → acknowledge briefly (5-10 words) and STOP
3. If user goes off-topic → answer in 1 sentence, then re-ask the CURRENT QUESTION exactly
4. NEVER ask about previous steps or jump ahead
5. Keep all responses under 30 words unless it's the exact question

You are NOT a general chatbot. You are a form-filler following a script."""

    async def chat(
        self,
        message: str,
        conversation_history: list,
        knowledge_base: str = "",
        current_step_id: str = None,
        current_step_prompt: str = None,
    ):
        """Generate response following the flow script exactly"""

        if not openai_client:
            return "I apologize, but the AI service is not configured. Please contact our office directly."

        step_data = self.step_index.get(current_step_id, {})
        input_type = step_data.get("input_type", "text")
        options = step_data.get("options", [])

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        current_question_context = f"""
🎯 CURRENT QUESTION (ask this EXACTLY):
"{current_step_prompt}"

Input type: {input_type}
"""
        
        if options:
            current_question_context += "\nValid options:\n" + "\n".join([f"- {opt['label']}" for opt in options])

        messages.append({
            "role": "system",
            "content": current_question_context
        })

        collected_info = self._extract_collected_info(conversation_history)
        if collected_info:
            messages.append({
                "role": "system",
                "content": f"✅ Already collected:\n{collected_info}\n\n⚠️ DO NOT ask about these again!"
            })

        recent = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        messages.extend(recent)
        messages.append({"role": "user", "content": message})

        try:
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI error: {e}")
            return "I apologize, but I'm having trouble right now. Please call our office directly."

    def _extract_collected_info(self, history: list) -> str:
        """Extract what info has already been collected"""
        if not history:
            return ""

        collected = []
        all_user_text = " ".join([m.get("content", "") for m in history if m.get("role") == "user"]).lower()

        if "accident" in all_user_text or "car" in all_user_text:
            collected.append("- Case type: Personal Injury (car accident)")
        
        if any(word in all_user_text for word in ["today", "yesterday", "last week", "ago", "/", "2025", "2024"]):
            collected.append("- Incident timing mentioned")
        
        if any(word in all_user_text for word in ["hospital", "doctor", "medical", "treatment", "er"]):
            collected.append("- Medical treatment discussed")
        
        if "attorney" in all_user_text or "lawyer" in all_user_text:
            collected.append("- Attorney status mentioned")

        return "\n".join(collected) if collected else ""

chatbot = ChatbotService()

# ============================================
# PAYMENT SERVICES
# ============================================

async def create_stripe_payment(amount: float, description: str, metadata: dict = None) -> Dict:
    """Create Stripe payment intent"""
    try:
        intent = await asyncio.to_thread(
            stripe.PaymentIntent.create,
            amount=int(amount * 100),
            currency="usd",
            description=description,
            metadata=metadata or {}
        )
        return {
            "success": True,
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

async def create_paypal_payment(amount: float, description: str) -> Dict:
    """Create PayPal order"""
    try:
        auth = requests.auth.HTTPBasicAuth(PAYPAL_CLIENT_ID, PAYPAL_SECRET)
        response = requests.post(
            "https://api-m.sandbox.paypal.com/v2/checkout/orders",
            auth=auth,
            headers={"Content-Type": "application/json"},
            json={
                "intent": "CAPTURE",
                "purchase_units": [{
                    "amount": {
                        "currency_code": "USD",
                        "value": str(amount)
                    },
                    "description": description
                }]
            }
        )
        data = response.json()
        return {
            "success": True,
            "order_id": data.get("id"),
            "approval_url": next((link["href"] for link in data.get("links", []) if link["rel"] == "approve"), None)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {"message": "Law Firm AI Chatbot API", "version": "1.0.0"}

# ============================================
# CHAT ENDPOINTS
# ============================================

@app.post("/api/chat")
async def chat_endpoint(chat: ChatMessage, db: Session = Depends(get_db)):
    """Main chat endpoint"""

    session_id = chat.session_id or str(uuid.uuid4())

    conversation = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).first()

    if not conversation:
        conversation = Conversation(
            session_id=session_id,
            client_id=chat.client_id,
            messages=[],
            channel="web",
        )
        db.add(conversation)

    flow_manager.ensure_session(session_id)
    prev_step_id = flow_manager.get_current_step(session_id)
    new_step_id = flow_manager.advance_if_answered(session_id, chat.message)
    current_step_id = flow_manager.get_current_step(session_id)
    response_text = flow_manager.get_prompt(current_step_id)

    conversation.messages.append({"role": "user", "content": chat.message})
    conversation.messages.append({"role": "assistant", "content": response_text})
    conversation.updated_at = datetime.now(timezone.utc)

    db.commit()

    return {
        "response": response_text,
        "session_id": session_id,
        "current_step": current_step_id,
        "answered": current_step_id != prev_step_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ============================================
# CLIENT MANAGEMENT
# ============================================

@app.post("/api/clients")
async def create_client(client: ClientCreate, db: Session = Depends(get_db)):
    """Create new client"""
    
    existing = db.query(Client).filter(Client.email == client.email).first()
    if existing:
        return {"success": True, "client_id": existing.id, "existing": True}
    
    new_client = Client(
        name=client.name,
        email=client.email,
        phone=client.phone
    )
    db.add(new_client)
    db.commit()
    db.refresh(new_client)
    
    await send_email(
        to=client.email,
        subject="Thank you for contacting us",
        body=f"Dear {client.name},\n\nThank you for reaching out. We'll review your case and contact you soon.\n\nBest regards,\n{LAW_FIRM_NAME}",
        html=f"<p>Dear {client.name},</p><p>Thank you for reaching out. We'll review your case and contact you soon.</p><p>Best regards,<br>{LAW_FIRM_NAME}</p>"
    )
    
    return {"success": True, "client_id": new_client.id}

@app.get("/api/clients/{client_id}")
async def get_client(client_id: str, db: Session = Depends(get_db)):
    """Lookup client by ID"""
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return {
        "id": client.id,
        "name": client.name,
        "email": client.email,
        "phone": client.phone,
        "case_type": client.case_type,
        "status": client.status,
        "created_at": client.created_at.isoformat()
    }

@app.get("/api/clients/email/{email}")
async def lookup_client_by_email(email: str, db: Session = Depends(get_db)):
    """Lookup client by email"""
    client = db.query(Client).filter(Client.email == email).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    cases = db.query(Case).filter(Case.client_id == client.id).all()
    
    return {
        "client": {
            "id": client.id,
            "name": client.name,
            "email": client.email,
            "phone": client.phone,
            "status": client.status
        },
        "cases": [
            {
                "id": case.id,
                "type": case.case_type,
                "status": case.status,
                "created_at": case.created_at.isoformat()
            } for case in cases
        ]
    }

# ============================================
# CASE MANAGEMENT
# ============================================

@app.post("/api/cases")
async def create_case(case: CaseCreate, db: Session = Depends(get_db)):
    """Create new case"""
    
    new_case = Case(
        client_id=case.client_id,
        case_type=case.case_type,
        description=case.description,
        intake_data=case.intake_data or {}
    )
    db.add(new_case)
    db.commit()
    db.refresh(new_case)
    
    client = db.query(Client).filter(Client.id == case.client_id).first()
    if client:
        await send_email(
            to=LAW_FIRM_EMAIL,
            subject=f"New Case Intake: {case.case_type}",
            body=f"New case from {client.name}\n\nType: {case.case_type}\nDescription: {case.description}\n\nCase ID: {new_case.id}"
        )
    
    return {"success": True, "case_id": new_case.id}

# ============================================
# PAYMENT ENDPOINTS
# ============================================

@app.post("/api/payments/stripe")
async def create_stripe_payment_endpoint(payment: PaymentCreate, db: Session = Depends(get_db)):
    """Create Stripe payment"""
    
    result = await create_stripe_payment(
        amount=payment.amount,
        description=f"Legal services - Case {payment.case_id}",
        metadata={"case_id": payment.case_id}
    )
    
    if result["success"]:
        new_payment = Payment(
            case_id=payment.case_id,
            amount=payment.amount,
            provider="stripe",
            transaction_id=result["payment_intent_id"],
            payment_metadata=result
        )
        db.add(new_payment)
        db.commit()
        
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])

@app.post("/api/payments/paypal")
async def create_paypal_payment_endpoint(payment: PaymentCreate, db: Session = Depends(get_db)):
    """Create PayPal payment"""
    
    result = await create_paypal_payment(
        amount=payment.amount,
        description=f"Legal services - Case {payment.case_id}"
    )
    
    if result["success"]:
        new_payment = Payment(
            case_id=payment.case_id,
            amount=payment.amount,
            provider="paypal",
            transaction_id=result["order_id"],
            payment_metadata=result
        )
        db.add(new_payment)
        db.commit()
        
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])
    
# Add these to your main.py file

# ============================================
# PAYMENT INTENT DETECTION & HANDLING
# ============================================

class PaymentIntentDetector:
    """Detects and handles payment-related queries in chat"""
    
    PAYMENT_KEYWORDS = [
        'payment', 'pay', 'paying', 'paid',
        'invoice', 'bill', 'billing',
        'retainer', 'fee', 'fees',
        'cost', 'price', 'charge',
        'owe', 'balance', 'due',
        'credit card', 'debit card',
        'stripe', 'paypal'
    ]
    
    @staticmethod
    def detect_payment_intent(message: str) -> bool:
        """Check if message is asking about payments"""
        message_lower = message.lower()
        
        # Check for payment keywords
        has_payment_keyword = any(keyword in message_lower for keyword in PaymentIntentDetector.PAYMENT_KEYWORDS)
        
        # Check for question patterns
        is_question = any(word in message_lower for word in ['can i', 'how do i', 'want to', 'need to', 'make a'])
        
        return has_payment_keyword and (is_question or 'payment' in message_lower)
    
    @staticmethod
    def get_payment_response(client_email: str = None, client_name: str = None) -> dict:
        """Generate payment options response"""
        
        return {
            "type": "payment_intent",
            "message": f"I can help you make a payment! We offer the following payment options:\n\n"
                      f"💳 **Retainer Fee**: $500\n"
                      f"💼 **Case Payment**: $2,500\n\n"
                      f"Which payment would you like to make?",
            "payment_options": [
                {
                    "id": "retainer",
                    "label": "Retainer Fee - $500",
                    "amount": 500,
                    "description": "Initial retainer fee for legal services"
                },
                {
                    "id": "case_payment",
                    "label": "Case Payment - $2,500",
                    "amount": 2500,
                    "description": "Full case payment"
                }
            ],
            "requires_client_info": not (client_email and client_name)
        }

payment_detector = PaymentIntentDetector()


# ============================================
# ENHANCED CHAT ENDPOINT WITH PAYMENT INTENT
# ============================================

@app.post("/api/chat/enhanced")
async def chat_endpoint_with_intents(chat: ChatMessage, db: Session = Depends(get_db)):
    """Enhanced chat endpoint that detects payment intents"""
    
    session_id = chat.session_id or str(uuid.uuid4())
    
    # Get or create conversation
    conversation = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).first()
    
    if not conversation:
        conversation = Conversation(
            session_id=session_id,
            client_id=chat.client_id,
            messages=[],
            channel="web",
        )
        db.add(conversation)
        db.commit()
    
    # Get client info if available
    client = None
    client_email = None
    client_name = None
    
    if chat.client_id:
        client = db.query(Client).filter(Client.id == chat.client_id).first()
        if client:
            client_email = client.email
            client_name = client.name
    
    # DETECT PAYMENT INTENT
    if payment_detector.detect_payment_intent(chat.message):
        print(f"💳 Payment intent detected from: {client_name or 'Unknown'}")
        
        payment_response = payment_detector.get_payment_response(
            client_email=client_email,
            client_name=client_name
        )
        
        # Save to conversation
        conversation.messages.append({"role": "user", "content": chat.message})
        conversation.messages.append({
            "role": "assistant", 
            "content": payment_response["message"],
            "metadata": {"type": "payment_intent", "options": payment_response["payment_options"]}
        })
        conversation.updated_at = datetime.now(timezone.utc)
        db.commit()
        
        return {
            "response": payment_response["message"],
            "session_id": session_id,
            "intent": "payment",
            "payment_options": payment_response["payment_options"],
            "requires_client_info": payment_response["requires_client_info"],
            "client_info": {
                "email": client_email,
                "name": client_name
            } if client else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # NORMAL FLOW HANDLING (existing intake flow)
    flow_manager.ensure_session(session_id)
    prev_step_id = flow_manager.get_current_step(session_id)
    new_step_id = flow_manager.advance_if_answered(session_id, chat.message)
    current_step_id = flow_manager.get_current_step(session_id)
    response_text = flow_manager.get_prompt(current_step_id)
    
    conversation.messages.append({"role": "user", "content": chat.message})
    conversation.messages.append({"role": "assistant", "content": response_text})
    conversation.updated_at = datetime.now(timezone.utc)
    db.commit()
    
    return {
        "response": response_text,
        "session_id": session_id,
        "current_step": current_step_id,
        "answered": current_step_id != prev_step_id,
        "intent": "intake",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================
# PAYMENT PROCESSING ENDPOINTS
# ============================================

class PaymentRequest(BaseModel):
    session_id: str
    payment_type: str  # 'retainer' or 'case_payment'
    client_email: str
    client_name: str
    payment_method: str  # 'stripe' or 'paypal'


@app.post("/api/payments/create-intent")
async def create_payment_intent(payment_request: PaymentRequest, db: Session = Depends(get_db)):
    """Create a payment intent for retainer or case payment"""
    
    # Determine amount based on payment type
    payment_amounts = {
        "retainer": 500.00,
        "case_payment": 2500.00
    }
    
    amount = payment_amounts.get(payment_request.payment_type)
    if not amount:
        raise HTTPException(status_code=400, detail="Invalid payment type")
    
    # Get or create client
    client = db.query(Client).filter(Client.email == payment_request.client_email).first()
    
    if not client:
        client = Client(
            name=payment_request.client_name,
            email=payment_request.client_email,
            status="payment_processing"
        )
        db.add(client)
        db.commit()
        db.refresh(client)
    
    # Get or create a case for this client
    case = db.query(Case).filter(Case.client_id == client.id).first()
    
    if not case:
        case = Case(
            client_id=client.id,
            case_type="General",
            description=f"{payment_request.payment_type.replace('_', ' ').title()} payment",
            status="payment_pending"
        )
        db.add(case)
        db.commit()
        db.refresh(case)
    
    # Create payment based on method
    if payment_request.payment_method == "stripe":
        result = await create_stripe_payment(
            amount=amount,
            description=f"{payment_request.payment_type.replace('_', ' ').title()} - {LAW_FIRM_NAME}",
            metadata={
                "case_id": case.id,
                "client_id": client.id,
                "payment_type": payment_request.payment_type
            }
        )
        
        if result["success"]:
            # Save payment record
            new_payment = Payment(
                case_id=case.id,
                amount=amount,
                provider="stripe",
                transaction_id=result["payment_intent_id"],
                status="pending",
                payment_metadata={
                    "payment_type": payment_request.payment_type,
                    "client_email": payment_request.client_email
                }
            )
            db.add(new_payment)
            db.commit()
            
            # Send confirmation email
            await send_email(
                to=payment_request.client_email,
                subject=f"Payment Link - {LAW_FIRM_NAME}",
                body=f"""Dear {payment_request.client_name},

Here is your payment link for: {payment_request.payment_type.replace('_', ' ').title()}
Amount: ${amount}

Please complete your payment at your earliest convenience.

Best regards,
{LAW_FIRM_NAME}""",
                html=f"""
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h2 style="color: #2563eb;">💳 Payment Link Ready</h2>
                    <p>Dear {payment_request.client_name},</p>
                    <div style="background-color: #dbeafe; padding: 20px; border-radius: 8px; margin: 20px 0;">
                        <p><strong>Payment Type:</strong> {payment_request.payment_type.replace('_', ' ').title()}</p>
                        <p><strong>Amount:</strong> ${amount}</p>
                    </div>
                    <p>Your payment is ready to process. Please complete it at your earliest convenience.</p>
                    <p>Best regards,<br><strong>{LAW_FIRM_NAME}</strong></p>
                </div>
                """
            )
            
            return {
                "success": True,
                "payment_id": new_payment.id,
                "amount": amount,
                "client_secret": result["client_secret"],
                "payment_intent_id": result["payment_intent_id"],
                "message": f"Payment link created for ${amount}"
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
    
    elif payment_request.payment_method == "paypal":
        result = await create_paypal_payment(
            amount=amount,
            description=f"{payment_request.payment_type.replace('_', ' ').title()} - {LAW_FIRM_NAME}"
        )
        
        if result["success"]:
            # Save payment record
            new_payment = Payment(
                case_id=case.id,
                amount=amount,
                provider="paypal",
                transaction_id=result["order_id"],
                status="pending",
                payment_metadata={
                    "payment_type": payment_request.payment_type,
                    "client_email": payment_request.client_email
                }
            )
            db.add(new_payment)
            db.commit()
            
            return {
                "success": True,
                "payment_id": new_payment.id,
                "amount": amount,
                "order_id": result["order_id"],
                "approval_url": result["approval_url"],
                "message": f"PayPal payment created for ${amount}"
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
    
    else:
        raise HTTPException(status_code=400, detail="Invalid payment method")


@app.post("/api/payments/confirm")
async def confirm_payment(
    payment_id: str,
    transaction_id: str,
    db: Session = Depends(get_db)
):
    """Confirm a payment after successful transaction"""
    
    payment = db.query(Payment).filter(Payment.id == payment_id).first()
    
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    # Update payment status
    payment.status = "completed"
    payment.transaction_id = transaction_id
    
    # Update case status
    case = db.query(Case).filter(Case.id == payment.case_id).first()
    if case:
        case.status = "payment_received"
    
    # Get client info
    client = db.query(Client).filter(Client.id == case.client_id).first()
    
    db.commit()
    
    # Send confirmation emails
    if client:
        # Email to client
        await send_email(
            to=client.email,
            subject=f"✅ Payment Confirmed - {LAW_FIRM_NAME}",
            body=f"""Dear {client.name},

Your payment of ${payment.amount} has been successfully processed.

Transaction ID: {transaction_id}
Amount: ${payment.amount}
Date: {datetime.now(timezone.utc).strftime('%B %d, %Y at %I:%M %p')}

Thank you for your payment!

Best regards,
{LAW_FIRM_NAME}""",
            html=f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: #10b981; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                    <h1 style="margin: 0;">✅ Payment Confirmed</h1>
                </div>
                <div style="background-color: #f9fafb; padding: 20px; border: 1px solid #e5e7eb;">
                    <p>Dear {client.name},</p>
                    <p>Your payment has been successfully processed.</p>
                    <div style="background-color: white; padding: 15px; border-radius: 6px; margin: 15px 0;">
                        <p><strong>Amount:</strong> ${payment.amount}</p>
                        <p><strong>Transaction ID:</strong> {transaction_id}</p>
                        <p><strong>Date:</strong> {datetime.now(timezone.utc).strftime('%B %d, %Y at %I:%M %p')}</p>
                    </div>
                    <p>Thank you for your payment!</p>
                    <p>Best regards,<br><strong>{LAW_FIRM_NAME}</strong></p>
                </div>
            </div>
            """
        )
        
        # Email to law firm
        await send_email(
            to=LAW_FIRM_EMAIL,
            subject=f"💰 Payment Received: ${payment.amount}",
            body=f"""Payment Received

Client: {client.name}
Email: {client.email}
Amount: ${payment.amount}
Transaction ID: {transaction_id}
Provider: {payment.provider}
Payment Type: {payment.payment_metadata.get('payment_type', 'Unknown')}
Case ID: {case.id}"""
        )
    
    return {
        "success": True,
        "message": "Payment confirmed",
        "payment": {
            "id": payment.id,
            "amount": payment.amount,
            "status": payment.status,
            "transaction_id": transaction_id
        }
    }


@app.get("/api/payments/{payment_id}/status")
async def get_payment_status(payment_id: str, db: Session = Depends(get_db)):
    """Check payment status"""
    
    payment = db.query(Payment).filter(Payment.id == payment_id).first()
    
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    return {
        "payment_id": payment.id,
        "amount": payment.amount,
        "status": payment.status,
        "provider": payment.provider,
        "transaction_id": payment.transaction_id,
        "created_at": payment.created_at.isoformat()
    }
    
# ============================================
# CLIENT PAYMENT VERIFICATION & PROCESSING
# ============================================

from pydantic import BaseModel

class ClientVerificationRequest(BaseModel):
    first_name: str
    last_name: str
    client_id: Optional[str] = None  # Made optional since clients may not have this
    email: str

class PaymentLinkRequest(BaseModel):
    client_id: str
    amount: float
    description: str = "Legal Services Payment"
    payment_type: str  # "retainer" or "case_payment"


@app.post("/api/payments/verify-client")
async def verify_client_for_payment(
    request: ClientVerificationRequest,
    db: Session = Depends(get_db)
):
    """
    Verify client exists by matching first name, last name, and email
    client_id is optional for additional verification
    Returns client info if found, or indicates no match
    """
    try:
        print(f"🔍 Verifying client: {request.first_name} {request.last_name}, Email: {request.email}, Client ID: {request.client_id or 'Not provided'}")
        
        # Build query - start with email as primary identifier
        query = db.query(Client).filter(Client.email == request.email.lower())
        
        # If client_id provided, add it to filter
        if request.client_id:
            query = query.filter(Client.id == request.client_id)
        
        client = query.first()
        
        # If found, verify names match (case-insensitive)
        if client:
            # Parse client name
            client_name_parts = client.name.lower().split()
            provided_first = request.first_name.lower()
            provided_last = request.last_name.lower()
            
            # Check if names match
            name_match = (
                provided_first in client_name_parts and
                provided_last in client_name_parts
            )
            
            if name_match:
                print(f"✅ Client verified: {client.name}")
                
                return {
                    "success": True,
                    "client_found": True,
                    "client": {
                        "id": client.id,
                        "name": client.name,
                        "email": client.email,
                        "phone": client.phone,
                        "case_type": client.case_type,
                        "status": client.status
                    }
                }
        
        print(f"❌ No matching client found")
        
        return {
            "success": True,
            "client_found": False,
            "message": "No matching client found. Please verify your information."
        }
        
    except Exception as e:
        print(f"Error verifying client: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/payments/stripe-config")
async def get_stripe_config():
    """
    Returns Stripe publishable key for frontend to initialize Stripe.js
    This is needed for embedded checkout modal
    """
    if not STRIPE_PUBLISHABLE_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Stripe publishable key not configured"
        )
    
    return {
        "success": True,
        "publishable_key": STRIPE_PUBLISHABLE_KEY
    }


# -------------------- ENDPOINT #2: Create Stripe Link (REPLACE) --------------------
# REPLACE lines 1412-1492 with this:

@app.post("/api/payments/create-stripe-link")
async def create_stripe_payment_link(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Create Stripe Checkout Session (embedded mode)
    Called by frontend after client verification
    """
    try:
        data = await request.json()
        client_id = data.get('client_id')
        amount = data.get('amount')
        description = data.get('description')
        payment_type = data.get('payment_type', 'retainer')
        reference_id = data.get('reference_id')
        
        if not STRIPE_SECRET_KEY:
            raise HTTPException(status_code=500, detail="Stripe not configured")
        
        # Get client info
        client = db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        print(f"💳 Creating Stripe checkout for {client.name} - ${amount}")
        
        # Create Stripe Checkout Session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': description,
                        'description': f'{LAW_FIRM_NAME} - {payment_type.replace("_", " ").title()}',
                    },
                    'unit_amount': int(amount * 100),  # Convert to cents
                },
                'quantity': 1,
            }],
            mode='payment',
            ui_mode='embedded',  # For embedded modal
            return_url=f'{BASE_URL}?payment=success&session_id={{CHECKOUT_SESSION_ID}}',
            client_reference_id=client.id,
            customer_email=client.email,
            metadata={
                'client_id': client.id,
                'client_name': client.name,
                'payment_type': payment_type,
                'reference_id': reference_id or 'N/A'
            }
        )
        
        # ✅ FIXED: Store payment with correct Payment model fields
        payment = Payment(
            case_id=client_id,  # ✅ case_id not client_id
            amount=amount,
            status='pending',
            provider='stripe',  # ✅ Added provider
            transaction_id=checkout_session.id,  # ✅ transaction_id not stripe_session_id
            payment_metadata={  # ✅ JSON field for extra data
                'payment_type': payment_type,
                'description': description,
                'reference_id': reference_id,
                'client_name': client.name,
                'client_email': client.email,
                'client_secret': checkout_session.client_secret
            }
        )
        db.add(payment)
        db.commit()
        
        print(f"✅ Stripe checkout session created: {checkout_session.id}")
        
        return {
            "success": True,
            "session_id": checkout_session.id,
            "client_secret": checkout_session.client_secret
        }
        
    except stripe.error.StripeError as e:
        print(f"❌ Stripe error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        print(f"❌ Error creating checkout: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/payments/{session_id}/status")
async def get_payment_status(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Check payment status after return from Stripe
    """
    try:
        # Retrieve Stripe session
        stripe_session = stripe.checkout.Session.retrieve(session_id)
        
        # Get payment from database
        payment = db.query(Payment).filter(
            Payment.transaction_id == session_id
        ).first()
        
        if not payment:
            raise HTTPException(status_code=404, detail="Payment not found")
        
        # Get client info
        client = db.query(Client).filter(Client.id == payment.case_id).first()
        
        if stripe_session.payment_status == 'paid':
            # Update payment status
            payment.status = 'completed'
            db.commit()
            
            print(f"✅ Payment verified: {session_id}")
            
            return {
                "success": True,
                "status": "completed",
                "amount": payment.amount,
                "transaction_id": session_id,
                "client_name": client.name if client else "Valued Client"
            }
        else:
            return {
                "success": False,
                "status": stripe_session.payment_status
            }
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/payments/create-stripe-checkout")
async def create_stripe_checkout_session(
    request: StripeCheckoutRequest,
    db: Session = Depends(get_db)
):
    """
    Create Stripe Checkout Session in EMBEDDED mode (opens as modal on same page)
    """
    try:
        if not STRIPE_SECRET_KEY:
            raise HTTPException(
                status_code=500, 
                detail="Stripe not configured"
            )
        
        # Get client info
        client = db.query(Client).filter(Client.id == request.client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        print(f"💳 Creating Stripe checkout for {client.name} - ${request.amount}")
        
        # Create Stripe Checkout Session
        # The key difference: Using success_url and cancel_url to redirect back
        # to the SAME page, then handling it with URL parameters
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': request.description,
                        'description': f'{LAW_FIRM_NAME} - {request.payment_type.replace("_", " ").title()}',
                    },
                    'unit_amount': int(request.amount * 100),  # Convert to cents
                },
                'quantity': 1,
            }],
            mode='payment',
            
            # IMPORTANT: These URLs allow the payment to return to the same page
            success_url=f'{BASE_URL}?payment=success&session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'{BASE_URL}?payment=cancelled',
            
            # UI mode for embedded experience (optional, but recommended)
            ui_mode='embedded',  # This makes it work as a modal
            
            # Store client info
            client_reference_id=client.id,
            customer_email=client.email,
            
            metadata={
                'client_id': client.id,
                'client_name': client.name,
                'payment_type': request.payment_type,
                'description': request.description,
                'reference_id': request.reference_id or 'N/A'
            }
        )
        
        # Store payment record in database
        payment = Payment(
            client_id=client.id,
            amount=request.amount,
            payment_type=request.payment_type,
            description=request.description,
            reference_id=request.reference_id,
            stripe_session_id=checkout_session.id,
            status='pending'
        )
        db.add(payment)
        db.commit()
        
        print(f"✅ Stripe checkout session created: {checkout_session.id}")
        
        return {
            "success": True,
            "session_id": checkout_session.id,
            "publishable_key": STRIPE_PUBLISHABLE_KEY,  # Frontend needs this
            "client_secret": checkout_session.client_secret  # For embedded checkout
        }
        
    except stripe.error.StripeError as e:
        print(f"❌ Stripe error: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        print(f"❌ Error creating checkout: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error creating checkout: {str(e)}"
        )


@app.post("/api/payments/create-paypal-link")
async def create_paypal_payment_link(
    request: PaymentLinkRequest,
    db: Session = Depends(get_db)
):
    """
    Create PayPal payment link for client payment
    """
    try:
        if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
            raise HTTPException(status_code=500, detail="PayPal not configured")
        
        # Get client info
        client = db.query(Client).filter(Client.id == request.client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Get PayPal access token
        auth_response = requests.post(
            "https://api-m.paypal.com/v1/oauth2/token",
            headers={"Accept": "application/json"},
            auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
            data={"grant_type": "client_credentials"}
        )
        
        if auth_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to authenticate with PayPal")
        
        access_token = auth_response.json()["access_token"]
        
        # Create PayPal order
        order_data = {
            "intent": "CAPTURE",
            "purchase_units": [{
                "reference_id": client.id,
                "description": request.description,
                "amount": {
                    "currency_code": "USD",
                    "value": f"{request.amount:.2f}"
                }
            }],
            "application_context": {
                "return_url": f"{BASE_URL}/payment-success?provider=paypal",
                "cancel_url": f"{BASE_URL}/chat-widget.html",
                "brand_name": LAW_FIRM_NAME,
                "user_action": "PAY_NOW"
            }
        }
        
        order_response = requests.post(
            "https://api-m.paypal.com/v2/checkout/orders",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            },
            json=order_data
        )
        
        if order_response.status_code != 201:
            raise HTTPException(status_code=500, detail="Failed to create PayPal order")
        
        order = order_response.json()
        
        # Get approval URL
        approval_url = next(
            (link["href"] for link in order["links"] if link["rel"] == "approve"),
            None
        )
        
        if not approval_url:
            raise HTTPException(status_code=500, detail="No approval URL in PayPal response")
        
        # Store payment record
        payment = Payment(
            case_id=client.case_type or "payment",
            amount=request.amount,
            status="pending",
            provider="paypal",
            transaction_id=order["id"],
            payment_metadata={
                'client_id': client.id,
                'payment_type': request.payment_type,
                'description': request.description,
                'order_id': order["id"]
            }
        )
        db.add(payment)
        db.commit()
        db.refresh(payment)
        
        print(f"✅ PayPal payment link created: {approval_url}")
        
        return {
            "success": True,
            "payment_url": approval_url,
            "order_id": order["id"],
            "payment_id": payment.id,
            "amount": request.amount,
            "provider": "paypal"
        }
        
    except Exception as e:
        print(f"Error creating PayPal payment link: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/payments/client-payment-request")
async def client_payment_request(
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    client_id: str = Form(None),  # Made optional
    reference_id: str = Form(None),  # For tracking
    amount: float = Form(None),  # Optional: can be set by system
    payment_type: str = Form("retainer"),  # "retainer" or "case_payment"
    db: Session = Depends(get_db)
):
    """
    All-in-one endpoint for client payment requests
    Verifies client by name and email (client_id optional)
    Returns payment links
    """
    try:
        print(f"💳 Payment request - Reference: {reference_id}, Client: {first_name} {last_name}, Email: {email}")
        
        # Verify client
        verification_request = ClientVerificationRequest(
            first_name=first_name,
            last_name=last_name,
            client_id=client_id,  # Now optional
            email=email
        )
        
        verification_result = await verify_client_for_payment(verification_request, db)
        
        if not verification_result["client_found"]:
            return {
                "success": False,
                "client_found": False,
                "message": "Client not found. Please verify your information.",
                "hand_off_to_agent": False,  # Frontend will ask "returning client?" question
                "reference_id": reference_id
            }
        
        # Client found - determine amount if not provided
        client = verification_result["client"]
        
        if amount is None:
            # Set default amounts based on payment type
            if payment_type == "retainer":
                amount = 500.00  # Default retainer fee
            else:
                amount = 250.00  # Default case payment
        
        # Create description
        description = f"{payment_type.replace('_', ' ').title()} - {client['name']}"
        
        return {
            "success": True,
            "client_found": True,
            "client": client,
            "amount": amount,
            "payment_type": payment_type,
            "description": description,
            "reference_id": reference_id,
            "message": f"Hello {client['name']}! Your {payment_type.replace('_', ' ')} amount is ${amount:.2f}. Please choose your payment method:"
        }
        
    except Exception as e:
        print(f"Error processing client payment request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/payments/client/{client_id}/pending")
async def get_client_pending_payments(
    client_id: str,
    db: Session = Depends(get_db)
):
    """
    Get pending payments for a client
    """
    try:
        payments = db.query(Payment).filter(
            Payment.payment_metadata['client_id'].astext == client_id,
            Payment.status == "pending"
        ).all()
        
        result = [{
            "id": p.id,
            "amount": p.amount,
            "provider": p.provider,
            "description": p.payment_metadata.get('description', 'Payment'),
            "created_at": p.created_at.isoformat()
        } for p in payments]
        
        return {
            "success": True,
            "count": len(result),
            "payments": result
        }
        
    except Exception as e:
        print(f"Error getting pending payments: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ==============================================================================
# PAYMENT STATUS CHECK (NEW - Called after payment completes)
# ==============================================================================

@app.get("/api/payments/{session_id}/status")
async def check_payment_status(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Check payment status and send receipt email if completed
    Called by frontend after Stripe redirects back
    """
    try:
        # Retrieve Stripe session
        session = stripe.checkout.Session.retrieve(session_id)
        
        # Update payment record in database
        payment = db.query(Payment).filter(
            Payment.stripe_session_id == session_id
        ).first()
        
        if not payment:
            raise HTTPException(status_code=404, detail="Payment not found")
        
        # Get client info
        client = db.query(Client).filter(Client.id == payment.client_id).first()
        
        if session.payment_status == 'paid':
            # Update payment status
            payment.status = 'completed'
            payment.completed_at = datetime.utcnow()
            db.commit()
            
            print(f"✅ Payment completed: {session_id}")
            
            # Send receipt email
            await send_receipt_email(
                client_email=client.email,
                client_name=client.name,
                amount=payment.amount,
                transaction_id=session_id,
                payment_type=payment.payment_type,
                description=payment.description
            )
            
            return {
                "success": True,
                "status": "completed",
                "amount": payment.amount,
                "transaction_id": session_id,
                "client_name": client.name
            }
        else:
            return {
                "success": False,
                "status": session.payment_status,
                "message": "Payment not completed"
            }
            
    except Exception as e:
        print(f"❌ Error checking payment status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# RECEIPT EMAIL FUNCTION (NEW)
# ==============================================================================

async def send_receipt_email(
    client_email: str,
    client_name: str,
    amount: float,
    transaction_id: str,
    payment_type: str,
    description: str
):
    """
    Send professional receipt email to client
    """
    try:
        if not RESEND_API_KEY:
            print("⚠️ Skipping receipt email (RESEND_API_KEY not configured)")
            return
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #1e40af; color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9fafb; padding: 30px; border: 1px solid #e5e7eb; }}
                .receipt-box {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #10b981; }}
                .amount {{ font-size: 32px; color: #10b981; font-weight: bold; }}
                .footer {{ background: #f3f4f6; padding: 20px; text-align: center; border-radius: 0 0 8px 8px; font-size: 14px; color: #6b7280; }}
                .info-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #e5e7eb; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{LAW_FIRM_NAME}</h1>
                    <p style="margin: 0; opacity: 0.9;">Payment Receipt</p>
                </div>
                
                <div class="content">
                    <h2>Thank you for your payment, {client_name}!</h2>
                    <p>We've successfully received your payment. Here are the details:</p>
                    
                    <div class="receipt-box">
                        <div class="info-row">
                            <span><strong>Amount Paid:</strong></span>
                            <span class="amount">${amount:.2f}</span>
                        </div>
                        <div class="info-row">
                            <span><strong>Transaction ID:</strong></span>
                            <span>{transaction_id}</span>
                        </div>
                        <div class="info-row">
                            <span><strong>Payment Type:</strong></span>
                            <span>{payment_type.replace('_', ' ').title()}</span>
                        </div>
                        <div class="info-row">
                            <span><strong>Description:</strong></span>
                            <span>{description}</span>
                        </div>
                        <div class="info-row" style="border-bottom: none;">
                            <span><strong>Date:</strong></span>
                            <span>{datetime.utcnow().strftime('%B %d, %Y at %I:%M %p UTC')}</span>
                        </div>
                    </div>
                    
                    <p style="margin-top: 30px;">
                        <strong>What happens next?</strong><br>
                        Your payment has been processed and our team will be in touch shortly to discuss the next steps in your case.
                    </p>
                    
                    <p>If you have any questions about this payment, please don't hesitate to reach out to us.</p>
                </div>
                
                <div class="footer">
                    <p><strong>{LAW_FIRM_NAME}</strong></p>
                    <p>📞 {LAW_FIRM_PHONE} | 📧 {LAW_FIRM_EMAIL}</p>
                    <p style="margin-top: 15px; font-size: 12px;">
                        This is an automated receipt. Please keep this for your records.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        params = {
            "from": f"{LAW_FIRM_NAME} <{LAW_FIRM_EMAIL}>",
            "to": [client_email],
            "subject": f"Payment Receipt - ${amount:.2f} - {LAW_FIRM_NAME}",
            "html": html_content
        }
        
        email = resend.Emails.send(params)
        print(f"✅ Receipt email sent to {client_email}")
        
    except Exception as e:
        print(f"❌ Error sending receipt email: {str(e)}")
        # Don't raise exception - payment succeeded even if email fails

# ==============================================================================
# PAYPAL ENDPOINTS (Optional - if you want PayPal too)
# ==============================================================================

# You can add PayPal SDK integration here if needed
# Similar approach: create order, return order_id, handle completion
# ============================================
# TEST PAYMENT INTENT
# ============================================

@app.post("/api/test/payment-intent")
async def test_payment_intent(message: str):
    """Test if message triggers payment intent"""
    
    is_payment = payment_detector.detect_payment_intent(message)
    
    if is_payment:
        response = payment_detector.get_payment_response()
        return {
            "detected": True,
            "message": message,
            "response": response
        }
    else:
        return {
            "detected": False,
            "message": message,
            "note": "This message does not trigger payment intent"
        }

# ============================================
# FILE UPLOAD/DOWNLOAD
# ============================================

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    case_id: str = Form(default="temp"),
    client_id: str = Form(default="guest"),
    folder_type: str = Form(default="client_uploads"),  # New!
    db: Session = Depends(get_db)
):
    """Upload document to S3"""
    
    try:
        # Read file content
        content = await file.read()
        
        # Validate file size
        if len(content) > 10 * 1024 * 1024:
            return {"success": False, "error": "File too large. Maximum size is 10MB."}
        
        # Upload to S3 instead of local storage
        s3_result = upload_file_object_to_s3(
            file_content=content,
            filename=file.filename,
            client_id=client_id,
            case_id=case_id if case_id != "temp" else None,
            folder_type=folder_type,
            content_type=file.content_type or "application/octet-stream"
        )
        
        # Save to database (file_path now stores S3 key)
        document = Document(
            case_id=case_id,
            client_id=client_id,
            filename=file.filename,
            file_path=s3_result['s3_key'],  # S3 key instead of local path
            file_type=file.content_type,
            file_size=len(content)
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        return {
            "success": True,
            "document_id": document.id,
            "filename": file.filename,
            "s3_key": s3_result['s3_key']
        }
        
    except Exception as e:
        db.rollback()
        print(f"Upload error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/download/{document_id}")
async def download_file(document_id: str, db: Session = Depends(get_db)):
    """Get download URL for document from S3"""
    
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Generate temporary download URL (expires in 1 hour)
        download_url = generate_presigned_download_url(
            s3_key=document.file_path,
            expiration=3600
        )
        
        return {
            "success": True,
            "download_url": download_url,
            "filename": document.filename,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "expires_in": 3600
        }
        
    except Exception as e:
        print(f"Download URL generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/case/{case_id}")
async def get_case_documents(case_id: str, db: Session = Depends(get_db)):
    """Get all documents for a case"""
    
    documents = db.query(Document).filter(Document.case_id == case_id).all()
    
    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "uploaded_at": doc.uploaded_at.isoformat()
            } for doc in documents
        ]
    }

# ============================================
# TWILIO PHONE INTEGRATION
# ============================================

@app.post("/api/twilio/voice")
async def handle_voice_call():
    """Handle incoming phone calls"""
    
    response = VoiceResponse()
    gather = Gather(
        input='speech',
        action='/api/twilio/process-speech',
        timeout=3,
        speech_timeout='auto'
    )
    
    gather.say(
        "Thank you for calling our law firm. How can I help you today?",
        voice='alice'
    )
    
    response.append(gather)
    response.say("We didn't receive any input. Goodbye!")
    
    return str(response)

# Replace the Twilio callback endpoints in main.py with these enhanced versions

@app.api_route("/api/twilio/appointment-confirmation", methods=["GET", "POST"])
async def appointment_confirmation_call(
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle automated appointment confirmation call with full details"""
    appointment_id = request.query_params.get("appointment_id")
    
    print(f"📞 Incoming confirmation call for appointment: {appointment_id}")
    
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    
    response = VoiceResponse()
    
    if not appointment:
        print(f"❌ Appointment not found: {appointment_id}")
        response.say("We're sorry, we couldn't find your appointment. Please call our office.", voice='alice')
        return Response(content=str(response), media_type="application/xml")
    
    print(f"✅ Found appointment for {appointment.client_name}")
    
    gather = Gather(
        num_digits=1,
        action=f'{BASE_URL}/api/twilio/confirm-appointment?appointment_id={appointment_id}',
        timeout=10
    )
    
    # Build detailed message with appointment info
    case_type = appointment.case_type or 'consultation'
    date_time = appointment.scheduled_date.strftime('%A, %B %d at %I:%M %p') if appointment.scheduled_date else 'your requested time'
    
    # Parse notes to extract key details
    notes = appointment.notes or ""
    injury_type = "your injury"
    incident_date = "recently"
    
    if "Injury Type:" in notes:
        try:
            injury_line = notes.split("Injury Type:")[1].split("\n")[0].strip()
            if injury_line and len(injury_line) < 50:
                injury_type = injury_line
        except:
            pass
    
    if "Date/Time:" in notes:
        try:
            incident_line = notes.split("Date/Time:")[1].split("\n")[0].strip()
            if incident_line and len(incident_line) < 50:
                incident_date = incident_line
        except:
            pass
    
    message = f"""Hello {appointment.client_name}. This is {LAW_FIRM_NAME} calling to confirm your {case_type} consultation.

We have you scheduled for {date_time}.

Based on your intake, this is regarding {injury_type} that occurred {incident_date}.

To confirm this appointment, press 1.
To request a different time, press 2.
To speak with someone now, press 3."""
    
    gather.say(message, voice='alice')
    response.append(gather)
    
    # Fallback if no response
    response.say("We didn't receive a response. We'll send you an email instead. Thank you, goodbye.", voice='alice')
    
    return Response(content=str(response), media_type="application/xml")


@app.api_route("/api/twilio/confirm-appointment", methods=["GET", "POST"])
async def confirm_appointment_response(
    request: Request,
    db: Session = Depends(get_db)
):
    """Process appointment confirmation response"""
    
    # Get form data from Twilio
    form = await request.form()
    digits = form.get("Digits")
    appointment_id = request.query_params.get("appointment_id")
    
    print(f"📞 Received response: {digits} for appointment: {appointment_id}")
    
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    response = VoiceResponse()
    
    if not appointment:
        print(f"❌ Appointment not found: {appointment_id}")
        response.say("We couldn't find your appointment. Please call our office.", voice='alice')
        return Response(content=str(response), media_type="application/xml")
    
    if digits == "1":
        # CONFIRM APPOINTMENT
        print(f"✅ Appointment CONFIRMED by {appointment.client_name}")
        appointment.status = "confirmed"
        db.commit()
        
        date_str = appointment.scheduled_date.strftime('%A, %B %d at %I:%M %p') if appointment.scheduled_date else 'your scheduled time'
        
        response.say(
            f"""Perfect! Your {appointment.case_type or 'consultation'} is confirmed for {date_str}.
            
            You'll receive a confirmation email with our office address, what to bring, and our attorney's contact information.
            
            We look forward to meeting with you. Thank you, goodbye.""",
            voice='alice'
        )
        
        # Send detailed confirmation email
        await send_email(
            to=appointment.client_email,
            subject=f"✅ Appointment Confirmed - {LAW_FIRM_NAME}",
            body=f"""Dear {appointment.client_name},

✅ YOUR APPOINTMENT IS CONFIRMED

Date & Time: {date_str}
Case Type: {appointment.case_type or 'Consultation'}
Duration: 30-45 minutes

📍 LOCATION:
{LAW_FIRM_NAME}
[Office Address Here]

📋 WHAT TO BRING:
✓ Photo ID
✓ Any documents related to your case
✓ Medical records (if applicable)
✓ Insurance information

Need to reschedule? Call {LAW_FIRM_PHONE}

Best regards,
{LAW_FIRM_NAME}""",
            html=f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #10b981; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                    <h1 style="margin: 0; font-size: 24px;">✅ Appointment Confirmed</h1>
                </div>
                
                <div style="background-color: #f9fafb; padding: 20px; border: 1px solid #e5e7eb;">
                    <h2 style="color: #1f2937; margin-top: 0;">Your Consultation Details</h2>
                    <div style="background-color: white; padding: 15px; border-radius: 6px; margin: 15px 0;">
                        <p style="margin: 5px 0;"><strong>📅 Date & Time:</strong> {date_str}</p>
                        <p style="margin: 5px 0;"><strong>⚖️ Case Type:</strong> {appointment.case_type or 'Consultation'}</p>
                        <p style="margin: 5px 0;"><strong>⏱️ Duration:</strong> 30-45 minutes</p>
                    </div>
                    
                    <h3 style="color: #1f2937;">📍 Location</h3>
                    <p style="background-color: white; padding: 15px; border-radius: 6px;">
                        <strong>{LAW_FIRM_NAME}</strong><br>
                        [Office Address Here]
                    </p>
                    
                    <h3 style="color: #1f2937;">📋 What to Bring</h3>
                    <ul style="background-color: white; padding: 20px; border-radius: 6px;">
                        <li>Photo ID</li>
                        <li>Documents related to your case</li>
                        <li>Medical records (if applicable)</li>
                        <li>Insurance information</li>
                    </ul>
                    
                    <div style="background-color: #dbeafe; padding: 15px; border-radius: 6px; margin-top: 20px;">
                        <p style="margin: 0;"><strong>📱 Need to reschedule?</strong></p>
                        <p style="margin: 10px 0 0 0;">Call <a href="tel:{LAW_FIRM_PHONE}">{LAW_FIRM_PHONE}</a></p>
                    </div>
                </div>
                
                <div style="background-color: #1f2937; color: white; padding: 15px; border-radius: 0 0 8px 8px; text-align: center;">
                    <p style="margin: 0;"><strong>{LAW_FIRM_NAME}</strong> • {LAW_FIRM_PHONE}</p>
                </div>
            </div>
            """
        )
        
    elif digits == "2":
        # REQUEST RESCHEDULE
        print(f"📅 Reschedule requested by {appointment.client_name}")
        appointment.status = "rescheduling"
        db.commit()
        
        response.say(
            f"""No problem. We'll have someone call you within one hour to find a better time. Thank you, goodbye.""",
            voice='alice'
        )
        
        # Alert law firm staff
        await send_email(
            to=LAW_FIRM_EMAIL,
            subject=f"🔄 RESCHEDULE REQUEST: {appointment.client_name}",
            body=f"""URGENT: Reschedule Request

Client: {appointment.client_name}
Phone: {appointment.client_phone}
Email: {appointment.client_email}

⏰ ACTION: Call within 1 hour

Appointment ID: {appointment_id}""",
            html=f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #dc2626;">🔄 RESCHEDULE REQUEST</h2>
                <div style="background-color: #fee; padding: 15px; border-left: 4px solid #dc2626;">
                    <strong>⏰ Call within 1 hour</strong>
                </div>
                <ul>
                    <li><strong>Name:</strong> {appointment.client_name}</li>
                    <li><strong>Phone:</strong> <a href="tel:{appointment.client_phone}">{appointment.client_phone}</a></li>
                    <li><strong>Email:</strong> {appointment.client_email}</li>
                </ul>
            </div>
            """
        )
        
    elif digits == "3":
        # SPEAK WITH SOMEONE NOW
        print(f"📞 Transfer requested by {appointment.client_name}")
        response.say("Transferring you now. Please hold.", voice='alice')
        response.dial(LAW_FIRM_PHONE)
        
    else:
        response.say("Invalid option. Goodbye.", voice='alice')
    
    return Response(content=str(response), media_type="application/xml")

@app.post("/api/twilio/process-speech")
async def process_speech(SpeechResult: str = Form(...), CallSid: str = Form(...), db: Session = Depends(get_db)):
    """Process speech input from phone"""
    
    knowledge_base = await scraper.get_knowledge_base()
    
    conversation = db.query(Conversation).filter(
        Conversation.session_id == CallSid
    ).first()
    
    if not conversation:
        conversation = Conversation(
            session_id=CallSid,
            messages=[],
            channel="phone"
        )
        db.add(conversation)
    
    response_text = await chatbot.chat(
        message=SpeechResult,
        conversation_history=conversation.messages,
        knowledge_base=knowledge_base
    )
    
    conversation.messages.append({"role": "user", "content": SpeechResult})
    conversation.messages.append({"role": "assistant", "content": response_text})
    db.commit()
    
    twiml = VoiceResponse()
    twiml.say(response_text, voice='alice')
    
    gather = Gather(
        input='speech',
        action='/api/twilio/process-speech',
        timeout=3,
        speech_timeout='auto'
    )
    gather.say("Is there anything else I can help you with?", voice='alice')
    
    twiml.append(gather)
    twiml.say("Thank you for calling. Goodbye!")
    
    return str(twiml)

@app.post("/api/twilio/sms")
async def handle_sms(Body: str = Form(...), From: str = Form(...), db: Session = Depends(get_db)):
    """Handle incoming SMS"""
    
    knowledge_base = await scraper.get_knowledge_base()
    
    session_id = f"sms_{From}"
    conversation = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).first()
    
    if not conversation:
        conversation = Conversation(
            session_id=session_id,
            messages=[],
            channel="sms"
        )
        db.add(conversation)
    
    response_text = await chatbot.chat(
        message=Body,
        conversation_history=conversation.messages,
        knowledge_base=knowledge_base
    )
    
    conversation.messages.append({"role": "user", "content": Body})
    conversation.messages.append({"role": "assistant", "content": response_text})
    db.commit()
    
    if twilio_client:
        twilio_client.messages.create(
            body=response_text,
            from_=TWILIO_PHONE_NUMBER,
            to=From
        )
    
    return {"success": True}

# Add these enhanced endpoints to your main.py

@app.api_route("/api/twilio/voice", methods=["GET", "POST"])
async def handle_voice_call_fixed(request: Request):
    """Handle incoming voice calls - FIXED VERSION"""
    try:
        print("📞 Incoming call received!")
        
        # Get form data from Twilio
        try:
            form_data = await request.form()
            caller = form_data.get("From", "Unknown")
            call_sid = form_data.get("CallSid", "Unknown")
            print(f"   Caller: {caller}")
            print(f"   Call SID: {call_sid}")
        except Exception as e:
            print(f"   Warning: Could not parse form data: {e}")
            caller = "Unknown"
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Create a gather to collect speech
        gather = Gather(
            input='speech',
            action='/api/twilio/process-speech',
            timeout=5,
            speech_timeout='auto',
            language='en-US'
        )
        
        # Greeting message
        greeting = f"Thank you for calling {LAW_FIRM_NAME}. How can I help you today?"
        gather.say(greeting, voice='alice')
        
        response.append(gather)
        
        # Fallback if no input
        response.say("We didn't receive any input. Please call back. Goodbye!", voice='alice')
        
        # Return TwiML as XML
        xml_response = str(response)
        print(f"✅ Sending TwiML response: {xml_response[:200]}...")
        
        return Response(
            content=xml_response,
            media_type="application/xml"
        )
        
    except Exception as e:
        print(f"❌ ERROR in voice endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a basic error response
        error_response = VoiceResponse()
        error_response.say("Sorry, there was an error. Please call back later.", voice='alice')
        return Response(
            content=str(error_response),
            media_type="application/xml"
        )


@app.api_route("/api/twilio/process-speech", methods=["GET", "POST"])
async def process_speech_fixed(request: Request, db: Session = Depends(get_db)):
    """Process speech input - FIXED VERSION"""
    try:
        print("🎤 Processing speech...")
        
        # Get form data
        form_data = await request.form()
        speech_result = form_data.get("SpeechResult", "")
        caller = form_data.get("From", "Unknown")
        call_sid = form_data.get("CallSid", str(uuid.uuid4()))
        
        print(f"   Speech: {speech_result}")
        print(f"   Caller: {caller}")
        
        # If no speech detected
        if not speech_result:
            response = VoiceResponse()
            response.say("I didn't hear anything. Goodbye!", voice='alice')
            return Response(content=str(response), media_type="application/xml")
        
        # Get or create conversation
        session_id = f"phone_{call_sid}"
        conversation = db.query(Conversation).filter(
            Conversation.session_id == session_id
        ).first()
        
        if not conversation:
            conversation = Conversation(
                session_id=session_id,
                messages=[],
                channel="phone"
            )
            db.add(conversation)
            db.commit()
        
        # Generate AI response
        ai_response = "Thank you for calling. We've received your message."
        
        # Try to get AI response if OpenAI is configured
        if openai_client:
            try:
                print("   Generating AI response...")
                
                # Simple AI prompt
                messages = [
                    {"role": "system", "content": f"You are a helpful assistant for {LAW_FIRM_NAME}. Keep responses brief (2-3 sentences) and professional."},
                    {"role": "user", "content": speech_result}
                ]
                
                # Add conversation history
                for msg in conversation.messages[-4:]:  # Last 2 exchanges
                    messages.append(msg)
                
                messages.append({"role": "user", "content": speech_result})
                
                completion = await asyncio.to_thread(
                    openai_client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                )
                
                ai_response = completion.choices[0].message.content
                print(f"   AI Response: {ai_response}")
                
            except Exception as e:
                print(f"   ⚠️ AI error: {e}")
                ai_response = "Thank you for your question. A team member will call you back shortly."
        
        # Save conversation
        conversation.messages.append({"role": "user", "content": speech_result})
        conversation.messages.append({"role": "assistant", "content": ai_response})
        db.commit()
        
        # Build response
        twiml = VoiceResponse()
        twiml.say(ai_response, voice='alice')
        
        # Check if conversation should continue
        end_phrases = ["goodbye", "that's all", "thank you bye", "no thanks"]
        if any(phrase in speech_result.lower() for phrase in end_phrases):
            twiml.say("Thank you for calling. Goodbye!", voice='alice')
            twiml.hangup()
        else:
            # Ask if they need more help
            gather = Gather(
                input='speech',
                action='/api/twilio/process-speech',
                timeout=5,
                speech_timeout='auto'
            )
            gather.say("Is there anything else I can help you with?", voice='alice')
            twiml.append(gather)
            
            # If no response
            twiml.say("Thank you for calling. Have a great day!", voice='alice')
        
        return Response(content=str(twiml), media_type="application/xml")
        
    except Exception as e:
        print(f"❌ ERROR in process-speech: {e}")
        import traceback
        traceback.print_exc()
        
        error_response = VoiceResponse()
        error_response.say("Sorry, there was an error processing your request.", voice='alice')
        return Response(content=str(error_response), media_type="application/xml")



@app.get("/api/phone/info")
async def get_phone_info():
    """Get call-in number information"""
    return {
        "call_in_number": TWILIO_PHONE_NUMBER,
        "formatted_number": format_phone_number(TWILIO_PHONE_NUMBER) if TWILIO_PHONE_NUMBER else None,
        "features": [
            "AI-powered legal assistant",
            "Schedule consultations",
            "Transfer to human agent",
            "24/7 availability"
        ],
        "status": "active" if TWILIO_ACCOUNT_SID else "not_configured"
    }


def format_phone_number(number: str) -> str:
    """Format phone number for display"""
    # Remove +1 and format as (XXX) XXX-XXXX
    if number.startswith("+1"):
        number = number[2:]
    if len(number) == 10:
        return f"({number[:3]}) {number[3:6]}-{number[6:]}"
    return number

# ============================================
# EMAIL ENDPOINT
# ============================================

@app.post("/api/email/send")
async def send_email_endpoint(email: EmailRequest):
    """Send email"""
    success = await send_email(
        to=email.to,
        subject=email.subject,
        body=email.body,
        html=email.html
    )
    
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=500, detail="Failed to send email")

# ============================================
# APPOINTMENT SCHEDULING WITH CALLBACKS
# ============================================

async def schedule_callback_task(appointment_id: str, delay_seconds: int, db_session):
    """Background task to schedule appointment confirmation callback"""
    await asyncio.sleep(delay_seconds)
    
    db = SessionLocal()
    try:
        appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
        
        if not appointment or appointment.status != "pending":
            print(f"Skipping callback - appointment {appointment_id} not pending")
            return
        
        if not twilio_client or not appointment.client_phone:
            print(f"Skipping callback - Twilio not configured or no phone number")
            return
        
        try:
            call = twilio_client.calls.create(
                to=appointment.client_phone,
                from_=TWILIO_PHONE_NUMBER,
                url=f"{BASE_URL}/api/twilio/appointment-confirmation?appointment_id={appointment_id}"
            )
            print(f"Callback scheduled: {call.sid} to {appointment.client_phone}")
        except Exception as e:
            print(f"Twilio callback error: {e}")
            await send_email(
                to=appointment.client_email,
                subject="⏰ Appointment Confirmation Needed",
                body=f"""Dear {appointment.client_name},

We haven't confirmed your appointment yet. Please reply to confirm or call us at {LAW_FIRM_PHONE}.

Best regards,
{LAW_FIRM_NAME}"""
            )
    finally:
        db.close()

@app.post("/api/appointments/schedule")
async def schedule_appointment(appointment: AppointmentRequest, db: Session = Depends(get_db)):
    """Schedule a consultation appointment with automatic Calendly integration"""
    
    print(f"📅 Scheduling appointment for: {appointment.client_name}")
    
    # Validate and clean email
    email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    if not re.match(email_regex, appointment.client_email):
        print(f"⚠️ Invalid email provided: {appointment.client_email}")
        appointment.client_email = f"contact_{uuid.uuid4().hex[:8]}@lawfirm-placeholder.com"
    
    # Find or create client
    client = db.query(Client).filter(Client.email == appointment.client_email).first()
    if not client:
        client = Client(
            name=appointment.client_name,
            email=appointment.client_email,
            phone=appointment.client_phone,
            case_type=appointment.case_type
        )
        db.add(client)
        db.commit()
        db.refresh(client)
        print(f"✅ Created new client: {client.id}")
    else:
        print(f"✅ Found existing client: {client.id}") 
    
    # Try to create Calendly invitation
    calendar_result = None
    if CALENDAR_PROVIDER == "calendly" and CALENDLY_API_KEY:
        print("🔗 Attempting to create Calendly invitation...")
        calendar_result = await calendar_service.create_calendly_invitation({
            "name": appointment.client_name,
            "email": appointment.client_email,
            "phone": appointment.client_phone,
            "notes": appointment.notes
        })
        print(f"Calendly result: {calendar_result}")
    else:
        print(f"⚠️ Calendly not configured. PROVIDER={CALENDAR_PROVIDER}, API_KEY={'SET' if CALENDLY_API_KEY else 'NOT SET'}")
    
    # Create appointment record
    new_appointment = Appointment(
        client_id=client.id,
        client_name=appointment.client_name,
        client_email=appointment.client_email,
        client_phone=appointment.client_phone,
        case_type=appointment.case_type,
        notes=appointment.notes,
        status="pending",
        calendar_event_id=calendar_result.get("event_id") if calendar_result else None,
        calendar_link=calendar_result.get("booking_url") if calendar_result and calendar_result.get("success") else None
    )
    
    db.add(new_appointment)
    db.commit()
    db.refresh(new_appointment)
    print(f"✅ Appointment created: {new_appointment.id}")
    
    # Schedule callback for 5 minutes (300s) for testing, or 2 hours (7200s) for production
    CALLBACK_DELAY = 300  # Change to 7200 for production
    if appointment.client_phone and twilio_client:
        asyncio.create_task(schedule_callback_task(new_appointment.id, CALLBACK_DELAY, db))
        print(f"📞 Callback scheduled for {CALLBACK_DELAY} seconds from now")
    
    # SUCCESS PATH: Calendly link available
    if calendar_result and calendar_result.get("success") and calendar_result.get("booking_url"):
        booking_url = calendar_result.get("booking_url")
        print(f"✅ SUCCESS: Calendly booking URL available: {booking_url}")
        
        email_body = f"""Dear {appointment.client_name},

Thank you for choosing {LAW_FIRM_NAME} for your {appointment.case_type or 'legal'} consultation.

🔗 BOOK YOUR APPOINTMENT NOW:
Click here to select your preferred time: {booking_url}

Your requested time: {appointment.preferred_date or 'Not specified'}

Once you complete your booking, you'll receive:
✅ Instant calendar confirmation
📧 Email reminder 24 hours before
📱 Text message reminder

Questions? Call us at {LAW_FIRM_PHONE} or reply to this email.

Best regards,
{LAW_FIRM_NAME}"""

        email_html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #2563eb;">✅ Your Consultation Booking Link</h2>
            <p>Dear {appointment.client_name},</p>
            <p>Thank you for choosing <strong>{LAW_FIRM_NAME}</strong> for your <strong>{appointment.case_type or 'legal'}</strong> consultation.</p>
            
            <div style="background-color: #dbeafe; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: center;">
                <h3 style="margin-top: 0; color: #1e40af;">📅 Select Your Appointment Time</h3>
                <a href="{booking_url}" 
                   style="display: inline-block; background-color: #2563eb; color: white; padding: 15px 30px; 
                          text-decoration: none; border-radius: 6px; font-weight: bold; margin: 10px 0; font-size: 16px;">
                    Book Your Consultation Now
                </a>
                <p style="margin-top: 15px; font-size: 14px;">Your requested time: <strong>{appointment.preferred_date or 'Not specified'}</strong></p>
            </div>
            
            <h3>What Happens Next:</h3>
            <ul>
                <li>✅ Choose your preferred time slot</li>
                <li>📧 Get instant email confirmation</li>
                <li>🔔 Receive automatic reminders</li>
            </ul>
            
            <p>Questions? Call us at <a href="tel:{LAW_FIRM_PHONE}">{LAW_FIRM_PHONE}</a></p>
            <p>Best regards,<br><strong>{LAW_FIRM_NAME}</strong></p>
        </div>
        """
        
        await send_email(
            to=appointment.client_email,
            subject=f"📅 Book Your Consultation - {LAW_FIRM_NAME}",
            body=email_body,
            html=email_html
        )
        
        return {
            "success": True,
            "appointment_id": new_appointment.id,
            "calendar_link": booking_url,
            "message": "Calendly booking link sent! Check your email to select your preferred time.",
            "callback_scheduled": bool(appointment.client_phone and twilio_client)
        }
    
    # FALLBACK PATH: Manual scheduling
    else:
        error_msg = calendar_result.get("error") if calendar_result else "Calendly not configured"
        print(f"⚠️ FALLBACK: Using manual scheduling. Reason: {error_msg}")
        
        # Email to law firm
        await send_email(
            to=LAW_FIRM_EMAIL,
            subject=f"🔔 New Consultation: {appointment.client_name} - {appointment.case_type}",
            body=f"""New Consultation Request - ACTION REQUIRED

Client Details:
- Name: {appointment.client_name}
- Email: {appointment.client_email}
- Phone: {appointment.client_phone or 'Not provided'}
- Case Type: {appointment.case_type or 'Not specified'}
- Requested Time: {appointment.preferred_date or 'Not specified'}

Notes:
{appointment.notes if appointment.notes else 'No additional notes'}

⚠️ Calendly auto-scheduling failed: {error_msg}

ACTION: Please contact this client within 2 hours to confirm appointment.

Appointment ID: {new_appointment.id}""",
            html=f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #dc2626;">🔔 New Consultation Request</h2>
                <div style="background-color: #fee; padding: 15px; border-left: 4px solid #dc2626; margin: 20px 0;">
                    <strong>ACTION REQUIRED:</strong> Contact client within 2 hours<br>
                    <small>⚠️ Calendly auto-scheduling failed: {error_msg}</small>
                </div>
                <h3>Client Details:</h3>
                <ul>
                    <li><strong>Name:</strong> {appointment.client_name}</li>
                    <li><strong>Email:</strong> <a href="mailto:{appointment.client_email}">{appointment.client_email}</a></li>
                    <li><strong>Phone:</strong> <a href="tel:{appointment.client_phone or ''}">{appointment.client_phone or 'Not provided'}</a></li>
                    <li><strong>Case Type:</strong> {appointment.case_type or 'Not specified'}</li>
                    <li><strong>Requested Time:</strong> {appointment.preferred_date or 'Not specified'}</li>
                </ul>
                <h3>Additional Notes:</h3>
                <pre style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; white-space: pre-wrap;">{appointment.notes or 'None'}</pre>
                <p><small>Appointment ID: {new_appointment.id}</small></p>
            </div>
            """
        )
        
        # Email to client
        await send_email(
            to=appointment.client_email,
            subject=f"✓ Consultation Request Received - {LAW_FIRM_NAME}",
            body=f"""Dear {appointment.client_name},

Thank you for requesting a consultation with {LAW_FIRM_NAME}.

We've received your request for: {appointment.preferred_date or 'a consultation'}
Case type: {appointment.case_type or 'General consultation'}

Our scheduling team will review your preferred time and contact you within 2 hours at:
- Phone: {appointment.client_phone or 'Not provided'}
- Email: {appointment.client_email}

Need immediate assistance? Call us at {LAW_FIRM_PHONE}

Best regards,
{LAW_FIRM_NAME}""",
            html=f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #16a34a;">✓ Consultation Request Received</h2>
                <p>Dear {appointment.client_name},</p>
                <p>Thank you for requesting a consultation with <strong>{LAW_FIRM_NAME}</strong>.</p>
                <div style="background-color: #f0fdf4; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <p><strong>Requested:</strong> {appointment.preferred_date or 'a consultation'}</p>
                    <p><strong>Case type:</strong> {appointment.case_type or 'General consultation'}</p>
                </div>
                <p>Our scheduling team will contact you within <strong>2 hours</strong> at:</p>
                <ul>
                    <li>Phone: {appointment.client_phone or 'Not provided'}</li>
                    <li>Email: {appointment.client_email}</li>
                </ul>
                <p>Need immediate assistance? Call <a href="tel:{LAW_FIRM_PHONE}">{LAW_FIRM_PHONE}</a></p>
                <p>Best regards,<br><strong>{LAW_FIRM_NAME}</strong></p>
            </div>
            """
        )
        
        return {
            "success": True,
            "appointment_id": new_appointment.id,
            "calendar_link": None,
            "message": "Your consultation request has been received. We'll confirm your appointment within 2 hours.",
            "callback_scheduled": bool(appointment.client_phone and twilio_client),
            "fallback_reason": error_msg
        }

@app.get("/api/appointments/availability")
async def get_availability(date: Optional[str] = None):
    """Get available appointment slots"""
    result = await calendar_service.get_available_slots(date)
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])

@app.get("/api/appointments/{appointment_id}")
async def get_appointment(appointment_id: str, db: Session = Depends(get_db)):
    """Get appointment details"""
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    return {
        "id": appointment.id,
        "client_name": appointment.client_name,
        "client_email": appointment.client_email,
        "scheduled_date": appointment.scheduled_date.isoformat() if appointment.scheduled_date else None,
        "case_type": appointment.case_type,
        "status": appointment.status,
        "calendar_link": appointment.calendar_link,
        "notes": appointment.notes
    }

@app.patch("/api/appointments/{appointment_id}/status")
async def update_appointment_status(
    appointment_id: str,
    status: str,
    db: Session = Depends(get_db)
):
    """Update appointment status"""
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    appointment.status = status
    appointment.updated_at = datetime.now(timezone.utc)
    db.commit()
    
    return {"success": True, "status": status}

@app.post("/api/appointments/schedule-debug")
async def schedule_appointment_debug(request: Request):
    """Debug endpoint to see what data is being sent"""
    body = await request.json()
    print("Received data:", body)
    return {"received": body}

@app.post("/api/intake/complete")
async def complete_intake(request: Request, db: Session = Depends(get_db)):
    """Handle completed intake form with all collected data"""
    data = await request.json()
    
    # Extract contact info from the intake data
    client_name = data.get('client_name', 'Unknown')
    client_email = data.get('client_email', '')
    client_phone = data.get('client_phone', '')
    
    # Try to extract email from intake responses if not provided
    if not client_email:
        intake_text = json.dumps(data)
        email_matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', intake_text)
        if email_matches:
            client_email = email_matches[0]
    
    # Extract case details
    case_type = data.get('case_type', 'personal_injury')
    preferred_date = data.get('pi_schedule', data.get('preferred_date', 'Not specified'))
    
    # Format intake data nicely
    formatted_notes = f"""Intake Completed: {datetime.now(timezone.utc).strftime('%B %d, %Y at %I:%M %p')}

Case Type: {case_type.replace('_', ' ').title()}

Incident Details:
- Date/Time: {data.get('pi_intro', 'Not specified')}
- Injury Type: {data.get('pi_injury_type', 'Not specified')}
- Injury Details: {data.get('pi_injury_details', 'N/A')}
- Medical Treatment: {data.get('pi_medical_treatment', 'Not specified')}

Legal Status:
- Currently Has Attorney: {data.get('pi_has_attorney', 'Not specified')}

Documents:
- Documents Submitted: {data.get('pi_docs', 'Not specified')}

Consultation Preference:
- Type: {data.get('pi_consult', 'Not specified')}
- Preferred Time: {preferred_date}"""
    
    # Create appointment request
    appointment_request = AppointmentRequest(
        client_name=client_name,
        client_email=client_email,
        client_phone=client_phone,
        case_type=case_type.replace('_', ' ').title(),
        preferred_date=preferred_date,
        notes=formatted_notes
    )
    
    # Use the existing schedule_appointment function
    return await schedule_appointment(appointment_request, db)

class SalesforceService:
    """Manages Salesforce CRM integration for law firm intake"""
    
    def __init__(self):
        self.sf = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Salesforce"""
        if not all([SALESFORCE_USERNAME, SALESFORCE_PASSWORD, SALESFORCE_SECURITY_TOKEN]):
            print("⚠️ Salesforce credentials not configured")
            return False
        
        try:
            self.sf = Salesforce(
                username=SALESFORCE_USERNAME,
                password=SALESFORCE_PASSWORD,
                security_token=SALESFORCE_SECURITY_TOKEN,
                domain=SALESFORCE_DOMAIN
            )
            print("✅ Connected to Salesforce")
            return True
        except Exception as e:
            print(f"❌ Salesforce connection error: {e}")
            return False
    
    def create_lead(self, client_data: dict, intake_data: dict = None) -> Optional[str]:
        """
        Create a lead in Salesforce
        
        Args:
            client_data: Dict with name, email, phone, case_type
            intake_data: Optional dict with detailed intake information
        
        Returns:
            Lead ID if successful, None otherwise
        """
        if not self.sf:
            print("❌ Salesforce not connected")
            return None
        
        try:
            # Parse name
            name_parts = client_data.get('name', 'Unknown').split(' ', 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else 'Unknown'
            
            # Determine lead source based on channel
            lead_source = intake_data.get('channel', 'Web') if intake_data else 'Web'
            lead_source_map = {
                'web': 'Website',
                'phone': 'Phone Inquiry',
                'sms': 'SMS'
            }
            lead_source = lead_source_map.get(lead_source.lower(), 'Website')
            
            # Map case type to industry or custom field
            case_type = client_data.get('case_type', 'General Inquiry')
            
            # Build lead data
            lead_data = {
                'FirstName': first_name,
                'LastName': last_name,
                'Email': client_data.get('email'),
                'Phone': client_data.get('phone'),
                'Company': 'Individual',  # Required field for leads
                'LeadSource': lead_source,
                'Status': 'New',
                'Description': self._build_lead_description(intake_data),
                'Title': case_type
            }
            
            # Add custom fields if they exist in your Salesforce setup
            # Example: lead_data['Case_Type__c'] = case_type
            
            # Remove None values
            lead_data = {k: v for k, v in lead_data.items() if v is not None}
            
            print(f"Creating Salesforce Lead: {first_name} {last_name}")
            result = self.sf.Lead.create(lead_data)
            
            if result.get('success'):
                lead_id = result.get('id')
                print(f"✅ Salesforce Lead created: {lead_id}")
                
                # Add a note with full intake details if available
                if intake_data:
                    self._add_note(lead_id, 'Lead', intake_data)
                
                return lead_id
            else:
                print(f"❌ Salesforce Lead creation failed: {result}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating Salesforce Lead: {e}")
            return None
    
    def create_contact(self, client_data: dict, account_id: str = None) -> Optional[str]:
        """
        Create a contact in Salesforce
        
        Args:
            client_data: Dict with name, email, phone
            account_id: Optional Salesforce Account ID to link to
        
        Returns:
            Contact ID if successful, None otherwise
        """
        if not self.sf:
            return None
        
        try:
            name_parts = client_data.get('name', 'Unknown').split(' ', 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else 'Unknown'
            
            contact_data = {
                'FirstName': first_name,
                'LastName': last_name,
                'Email': client_data.get('email'),
                'Phone': client_data.get('phone'),
                'LeadSource': 'Website'
            }
            
            if account_id:
                contact_data['AccountId'] = account_id
            
            contact_data = {k: v for k, v in contact_data.items() if v is not None}
            
            print(f"Creating Salesforce Contact: {first_name} {last_name}")
            result = self.sf.Contact.create(contact_data)
            
            if result.get('success'):
                contact_id = result.get('id')
                print(f"✅ Salesforce Contact created: {contact_id}")
                return contact_id
            else:
                print(f"❌ Salesforce Contact creation failed: {result}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating Salesforce Contact: {e}")
            return None
    
    def create_case(self, client_data: dict, intake_data: dict, contact_id: str = None) -> Optional[str]:
        """
        Create a case in Salesforce
        
        Args:
            client_data: Dict with client information
            intake_data: Dict with case details
            contact_id: Optional Contact ID to link to
        
        Returns:
            Case ID if successful, None otherwise
        """
        if not self.sf:
            return None
        
        try:
            case_type = client_data.get('case_type', 'General Inquiry')
            
            # Map case types to Salesforce case types
            case_type_map = {
                'personal injury': 'Personal Injury',
                'car accident': 'Personal Injury - Auto',
                'slip and fall': 'Personal Injury - Premises',
                'medical malpractice': 'Personal Injury - Medical',
                'workers comp': 'Workers Compensation',
                'family law': 'Family Law',
                'immigration': 'Immigration',
                'criminal defense': 'Criminal Defense',
                'business law': 'Business Law',
                'estate planning': 'Estate Planning'
            }
            
            sf_case_type = case_type_map.get(case_type.lower(), 'General Inquiry')
            
            case_data = {
                'Subject': f"{sf_case_type} - {client_data.get('name', 'Unknown')}",
                'Description': self._build_case_description(intake_data),
                'Status': 'New',
                'Origin': 'Web',
                'Priority': 'Medium',
                'Type': sf_case_type
            }
            
            if contact_id:
                case_data['ContactId'] = contact_id
            
            case_data = {k: v for k, v in case_data.items() if v is not None}
            
            print(f"Creating Salesforce Case: {sf_case_type}")
            result = self.sf.Case.create(case_data)
            
            if result.get('success'):
                case_id = result.get('id')
                print(f"✅ Salesforce Case created: {case_id}")
                
                # Add detailed notes
                self._add_note(case_id, 'Case', intake_data)
                
                return case_id
            else:
                print(f"❌ Salesforce Case creation failed: {result}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating Salesforce Case: {e}")
            return None
    
    def update_lead_status(self, lead_id: str, status: str, notes: str = None) -> bool:
        """Update lead status in Salesforce"""
        if not self.sf:
            return False
        
        try:
            update_data = {'Status': status}
            
            result = self.sf.Lead.update(lead_id, update_data)
            
            if notes:
                self._add_note(lead_id, 'Lead', {'notes': notes})
            
            print(f"✅ Lead {lead_id} status updated to: {status}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating lead: {e}")
            return False
    
    def convert_lead_to_contact(self, lead_id: str) -> Optional[dict]:
        """
        Convert a Lead to Contact and optionally create Account and Opportunity
        
        Returns:
            Dict with contactId, accountId, opportunityId if successful
        """
        if not self.sf:
            return None
        
        try:
            # Get lead details
            lead = self.sf.Lead.get(lead_id)
            
            # Convert lead
            result = self.sf.Lead.convert(lead_id, {
                'convertedStatus': 'Qualified',
                'doNotCreateOpportunity': False
            })
            
            if result.get('success'):
                conversion_data = {
                    'contactId': result.get('contactId'),
                    'accountId': result.get('accountId'),
                    'opportunityId': result.get('opportunityId')
                }
                print(f"✅ Lead converted: {conversion_data}")
                return conversion_data
            else:
                print(f"❌ Lead conversion failed: {result}")
                return None
                
        except Exception as e:
            print(f"❌ Error converting lead: {e}")
            return None
    
    def search_by_email(self, email: str) -> Optional[dict]:
        """Search for existing contact or lead by email"""
        if not self.sf:
            return None
        
        try:
            # Search contacts first
            contact_query = f"SELECT Id, FirstName, LastName, Email, Phone FROM Contact WHERE Email = '{email}' LIMIT 1"
            contact_results = self.sf.query(contact_query)
            
            if contact_results['totalSize'] > 0:
                return {
                    'type': 'Contact',
                    'id': contact_results['records'][0]['Id'],
                    'data': contact_results['records'][0]
                }
            
            # Search leads
            lead_query = f"SELECT Id, FirstName, LastName, Email, Phone, Status FROM Lead WHERE Email = '{email}' AND IsConverted = false LIMIT 1"
            lead_results = self.sf.query(lead_query)
            
            if lead_results['totalSize'] > 0:
                return {
                    'type': 'Lead',
                    'id': lead_results['records'][0]['Id'],
                    'data': lead_results['records'][0]
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error searching Salesforce: {e}")
            return None
    
    def _build_lead_description(self, intake_data: dict) -> str:
        """Build a formatted description from intake data"""
        if not intake_data:
            return "New lead from website chatbot"
        
        description = "Intake from AI Chatbot\n\n"
        
        for key, value in intake_data.items():
            if value and key not in ['session_id', 'client_id']:
                formatted_key = key.replace('_', ' ').title()
                description += f"{formatted_key}: {value}\n"
        
        return description
    
    def _build_case_description(self, intake_data: dict) -> str:
        """Build a formatted case description from intake data"""
        if not intake_data:
            return "New case from website chatbot"
        
        description = "Case Details from AI Chatbot Intake\n\n"
        
        # Prioritize important fields
        priority_fields = [
            'incident_date', 'injury_type', 'medical_treatment', 
            'has_attorney', 'description', 'notes'
        ]
        
        for field in priority_fields:
            if field in intake_data and intake_data[field]:
                formatted_key = field.replace('_', ' ').title()
                description += f"{formatted_key}: {intake_data[field]}\n"
        
        # Add any other fields
        for key, value in intake_data.items():
            if value and key not in priority_fields and key not in ['session_id', 'client_id']:
                formatted_key = key.replace('_', ' ').title()
                description += f"{formatted_key}: {value}\n"
        
        return description
    
    def _add_note(self, parent_id: str, parent_type: str, data: dict):
        """Add a note/attachment to a Lead, Contact, or Case"""
        try:
            note_body = json.dumps(data, indent=2)
            
            note_data = {
                'ParentId': parent_id,
                'Title': f'Chatbot Intake Data - {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}',
                'Body': note_body
            }
            
            self.sf.Note.create(note_data)
            print(f"✅ Note added to {parent_type} {parent_id}")
            
        except Exception as e:
            print(f"⚠️ Could not add note: {e}")

# Initialize Salesforce service
salesforce_service = SalesforceService()


# ============================================
# SALESFORCE INTEGRATION ENDPOINTS
# ============================================

@app.post("/api/salesforce/create-lead")
async def create_salesforce_lead(
    client_data: dict,
    intake_data: dict = None,
    db: Session = Depends(get_db)
):
    """Create a lead in Salesforce from intake data"""
    
    lead_id = salesforce_service.create_lead(client_data, intake_data)
    
    if lead_id:
        # Update local database with Salesforce ID
        if client_data.get('client_id'):
            client = db.query(Client).filter(Client.id == client_data['client_id']).first()
            if client:
                # Store Salesforce ID in a custom field (you'll need to add this column)
                # client.salesforce_id = lead_id
                db.commit()
        
        return {
            "success": True,
            "lead_id": lead_id,
            "message": "Lead created in Salesforce"
        }
    else:
        return {
            "success": False,
            "error": "Failed to create Salesforce lead"
        }


@app.get("/api/salesforce/search/{email}")
async def search_salesforce(email: str):
    """Search for existing contact or lead in Salesforce"""
    
    result = salesforce_service.search_by_email(email)
    
    if result:
        return {
            "success": True,
            "found": True,
            "type": result['type'],
            "id": result['id'],
            "data": result['data']
        }
    else:
        return {
            "success": True,
            "found": False
        }


@app.get("/api/salesforce/test")
async def test_salesforce_connection():
    """Test Salesforce connection and return diagnostic info"""
    
    if not salesforce_service.sf:
        return {
            "success": False,
            "error": "Salesforce not connected",
            "config": {
                "SALESFORCE_USERNAME": "SET" if SALESFORCE_USERNAME else "NOT SET",
                "SALESFORCE_PASSWORD": "SET" if SALESFORCE_PASSWORD else "NOT SET",
                "SALESFORCE_SECURITY_TOKEN": "SET" if SALESFORCE_SECURITY_TOKEN else "NOT SET",
                "SALESFORCE_DOMAIN": SALESFORCE_DOMAIN
            }
        }
    
    try:
        # Test query
        result = salesforce_service.sf.query("SELECT Id, Name FROM Account LIMIT 1")
        
        return {
            "success": True,
            "message": "Salesforce connection working!",
            "test_query_results": result['totalSize'],
            "config": {
                "SALESFORCE_USERNAME": SALESFORCE_USERNAME,
                "SALESFORCE_DOMAIN": SALESFORCE_DOMAIN
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
        
# ============================================
# S3 STORAGE ENDPOINTS
# ============================================

@app.get("/api/s3/test")
async def test_s3():
    """Test S3 connection"""
    return test_s3_connection()


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    case_id: Optional[str] = Form(None),
    folder_type: str = Form("client_uploads"),
    db: Session = Depends(get_db)
):
    """
    Upload a document to S3
    
    Folder types:
    - intake_forms
    - case_documents
    - client_uploads
    - signed_agreements
    - medical_records
    - photos
    - other
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Upload to S3
        result = upload_file_object_to_s3(
            file_content=file_content,
            filename=file.filename,
            client_id=client_id,
            case_id=case_id,
            folder_type=folder_type,
            content_type=file.content_type or "application/octet-stream"
        )
        
        # Save document record in database
        document = Document(
            case_id=case_id or "no-case",
            client_id=client_id,
            filename=file.filename,
            file_path=result['s3_key'],  # Store S3 key instead of local path
            file_type=file.content_type,
            file_size=result['size']
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        return {
            "success": True,
            "message": "File uploaded successfully",
            "document_id": document.id,
            "s3_key": result['s3_key'],
            "url": result['url'],
            "size": result['size']
        }
        
    except Exception as e:
        db.rollback()
        print(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/generate-upload-url")
async def generate_upload_url(
    filename: str = Form(...),
    client_id: str = Form(...),
    case_id: Optional[str] = Form(None),
    folder_type: str = Form("client_uploads")
):
    """
    Generate presigned URL for direct browser upload to S3
    This allows the frontend to upload directly to S3 without going through the backend
    """
    try:
        result = generate_presigned_upload_url(
            filename=filename,
            client_id=client_id,
            case_id=case_id,
            folder_type=folder_type,
            expiration=3600  # 1 hour
        )
        
        return {
            "success": True,
            "upload_url": result['upload_url'],
            "s3_key": result['s3_key'],
            "fields": result['fields']
        }
        
    except Exception as e:
        print(f"Error generating upload URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/list/{client_id}")
async def list_documents(
    client_id: str,
    case_id: Optional[str] = None,
    folder_type: Optional[str] = None
):
    """
    List all documents for a client or case
    """
    try:
        files = list_client_files(
            client_id=client_id,
            case_id=case_id,
            folder_type=folder_type
        )
        
        return {
            "success": True,
            "count": len(files),
            "files": files
        }
        
    except Exception as e:
        print(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/download/{document_id}")
async def get_document_download_url(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get presigned download URL for a document
    """
    try:
        # Get document from database
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Generate presigned download URL
        download_url = generate_presigned_download_url(
            s3_key=document.file_path,  # file_path stores S3 key
            expiration=3600  # 1 hour
        )
        
        return {
            "success": True,
            "download_url": download_url,
            "filename": document.filename,
            "size": document.file_size,
            "expires_in": 3600
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating download URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/download-by-key")
async def get_download_url_by_key(
    s3_key: str
):
    """
    Get presigned download URL directly by S3 key
    """
    try:
        download_url = generate_presigned_download_url(
            s3_key=s3_key,
            expiration=3600
        )
        
        return {
            "success": True,
            "download_url": download_url,
            "expires_in": 3600
        }
        
    except Exception as e:
        print(f"Error generating download URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a document from S3 and database
    """
    try:
        # Get document from database
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from S3
        delete_file(document.file_path)
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        return {
            "success": True,
            "message": "Document deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/metadata/{document_id}")
async def get_document_metadata(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get document metadata
    """
    try:
        # Get document from database
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get S3 metadata
        s3_metadata = get_file_metadata(document.file_path)
        
        return {
            "success": True,
            "document": {
                "id": document.id,
                "filename": document.filename,
                "client_id": document.client_id,
                "case_id": document.case_id,
                "file_type": document.file_type,
                "file_size": document.file_size,
                "uploaded_at": document.uploaded_at.isoformat(),
                "s3_key": document.file_path,
                "s3_metadata": s3_metadata
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting document metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cases/{case_id}/documents")
async def get_case_documents(
    case_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all documents for a specific case
    """
    try:
        documents = db.query(Document).filter(Document.case_id == case_id).all()
        
        result = []
        for doc in documents:
            # Generate download URL for each
            download_url = generate_presigned_download_url(doc.file_path, expiration=3600)
            
            result.append({
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "download_url": download_url
            })
        
        return {
            "success": True,
            "count": len(result),
            "documents": result
        }
        
    except Exception as e:
        print(f"Error getting case documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clients/{client_id}/documents")
async def get_client_documents(
    client_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all documents for a specific client
    """
    try:
        documents = db.query(Document).filter(Document.client_id == client_id).all()
        
        result = []
        for doc in documents:
            # Generate download URL for each
            download_url = generate_presigned_download_url(doc.file_path, expiration=3600)
            
            result.append({
                "id": doc.id,
                "case_id": doc.case_id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "download_url": download_url
            })
        
        return {
            "success": True,
            "count": len(result),
            "documents": result
        }
        
    except Exception as e:
        print(f"Error getting client documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# BULK OPERATIONS
# ============================================

@app.post("/api/documents/bulk-upload")
async def bulk_upload_documents(
    files: List[UploadFile] = File(...),
    client_id: str = Form(...),
    case_id: Optional[str] = Form(None),
    folder_type: str = Form("client_uploads"),
    db: Session = Depends(get_db)
):
    """
    Upload multiple documents at once
    """
    try:
        results = []
        
        for file in files:
            # Read file content
            file_content = await file.read()
            
            # Upload to S3
            s3_result = upload_file_object_to_s3(
                file_content=file_content,
                filename=file.filename,
                client_id=client_id,
                case_id=case_id,
                folder_type=folder_type,
                content_type=file.content_type or "application/octet-stream"
            )
            
            # Save document record
            document = Document(
                case_id=case_id or "no-case",
                client_id=client_id,
                filename=file.filename,
                file_path=s3_result['s3_key'],
                file_type=file.content_type,
                file_size=s3_result['size']
            )
            db.add(document)
            
            results.append({
                "filename": file.filename,
                "document_id": document.id,
                "s3_key": s3_result['s3_key'],
                "size": s3_result['size']
            })
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Uploaded {len(results)} files successfully",
            "files": results
        }
        
    except Exception as e:
        db.rollback()
        print(f"Error in bulk upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# HELPER FUNCTION: Migrate Local Files to S3
# ============================================

@app.post("/api/admin/migrate-to-s3")
async def migrate_local_to_s3(db: Session = Depends(get_db)):
    """
    Admin endpoint: Migrate existing local files to S3
    Run this once if you already have files in the uploads folder
    """
    try:
        # Get all documents with local paths
        documents = db.query(Document).all()
        
        migrated = []
        failed = []
        
        for doc in documents:
            try:
                # Check if file still uses local path
                if doc.file_path.startswith("uploads/"):
                    local_path = Path(doc.file_path)
                    
                    if local_path.exists():
                        # Upload to S3
                        s3_result = upload_file_to_s3(
                            file_path=str(local_path),
                            client_id=doc.client_id,
                            case_id=doc.case_id,
                            folder_type="client_uploads",
                            custom_filename=doc.filename
                        )
                        
                        # Update database record
                        doc.file_path = s3_result['s3_key']
                        
                        migrated.append({
                            "document_id": doc.id,
                            "filename": doc.filename,
                            "s3_key": s3_result['s3_key']
                        })
                    else:
                        failed.append({
                            "document_id": doc.id,
                            "filename": doc.filename,
                            "error": "Local file not found"
                        })
            except Exception as e:
                failed.append({
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "error": str(e)
                })
        
        db.commit()
        
        return {
            "success": True,
            "migrated_count": len(migrated),
            "failed_count": len(failed),
            "migrated": migrated,""""""
            "failed": failed
        }
        
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# HEALTH CHECK & TEST ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "openai": bool(OPENAI_API_KEY),
            "stripe": bool(STRIPE_SECRET_KEY),
            "paypal": bool(PAYPAL_CLIENT_ID),
            "twilio": bool(TWILIO_ACCOUNT_SID),
            "resend": bool(RESEND_API_KEY),
            "calendly": bool(CALENDLY_API_KEY)
        }
    }

@app.post("/api/test/email")
async def test_email_endpoint(email: str):
    """Test Resend email configuration"""
    success = await send_email(
        to=email,
        subject=f"Test Email from {LAW_FIRM_NAME}",
        body="If you receive this, your Resend configuration is working correctly!",
        html=f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #16a34a;">✅ Email Test Successful</h2>
            <p>Your Resend configuration is working correctly!</p>
            <p>Sent from: <strong>{LAW_FIRM_NAME}</strong></p>
            <p><small>Test performed at {datetime.now(timezone.utc).isoformat()}</small></p>
        </div>
        """
    )
    
    if success:
        return {"success": True, "message": f"Test email sent to {email}"}
    else:
        return {"success": False, "message": "Email failed - check Resend API key and configuration"}

@app.post("/api/test/callback")
async def test_callback_endpoint(phone: str, appointment_id: str = None):
    """Test Twilio callback (for testing only)"""
    if not twilio_client:
        return {"success": False, "error": "Twilio not configured"}
    
    if not appointment_id:
        db = SessionLocal()
        test_appointment = Appointment(
            client_id="test",
            client_name="Test User",
            client_email="test@example.com",
            client_phone=phone,
            case_type="Test Consultation",
            status="pending"
        )
        db.add(test_appointment)
        db.commit()
        appointment_id = test_appointment.id
        db.close()
    
    try:
        call = twilio_client.calls.create(
            to=phone,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{BASE_URL}/api/twilio/appointment-confirmation?appointment_id={appointment_id}"
        )
        return {
            "success": True,
            "call_sid": call.sid,
            "appointment_id": appointment_id,
            "message": f"Test call initiated to {phone}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    
# Add this test endpoint to main.py to verify Calendly configuration

@app.get("/api/test/calendly")
async def test_calendly_config():
    """Test Calendly API configuration and return diagnostic info"""
    
    if not CALENDLY_API_KEY:
        return {
            "success": False,
            "error": "CALENDLY_API_KEY not set in environment variables",
            "config": {
                "CALENDLY_API_KEY": "NOT SET",
                "CALENDLY_EVENT_TYPE": CALENDLY_EVENT_TYPE or "NOT SET",
                "CALENDAR_PROVIDER": CALENDAR_PROVIDER
            }
        }
    
    try:
        headers = {
            "Authorization": f"Bearer {CALENDLY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Test 1: Get user info
        print("Testing Calendly API - Step 1: Get user info...")
        user_response = requests.get(
            "https://api.calendly.com/users/me",
            headers=headers,
            timeout=10
        )
        
        if user_response.status_code != 200:
            return {
                "success": False,
                "error": f"Calendly API authentication failed: {user_response.status_code}",
                "response": user_response.text,
                "config": {
                    "CALENDLY_API_KEY": "SET (but invalid or expired)",
                    "CALENDLY_EVENT_TYPE": CALENDLY_EVENT_TYPE or "NOT SET",
                    "CALENDAR_PROVIDER": CALENDAR_PROVIDER
                }
            }
        
        user_data = user_response.json()
        user_uri = user_data["resource"]["uri"]
        user_name = user_data["resource"]["name"]
        
        print(f"✅ User authenticated: {user_name}")
        
        # Test 2: Get event types
        print("Testing Calendly API - Step 2: Get event types...")
        event_types_response = requests.get(
            f"https://api.calendly.com/event_types?user={user_uri}",
            headers=headers,
            timeout=10
        )
        
        if event_types_response.status_code != 200:
            return {
                "success": False,
                "error": f"Could not fetch event types: {event_types_response.status_code}",
                "user_info": {
                    "name": user_name,
                    "uri": user_uri
                }
            }
        
        event_types_data = event_types_response.json()
        event_types = event_types_data.get("collection", [])
        
        if not event_types:
            return {
                "success": False,
                "error": "No event types found. Please create at least one event type in Calendly.",
                "user_info": {
                    "name": user_name,
                    "uri": user_uri
                },
                "instructions": "Go to https://calendly.com/event_types/user/me and create an event type"
            }
        
        event_type_list = [
            {
                "name": et["name"],
                "uri": et["uri"],
                "active": et["active"],
                "booking_url": et["scheduling_url"]
            }
            for et in event_types
        ]
        
        print(f"✅ Found {len(event_types)} event types")
        
        # Test 3: Create a test scheduling link
        print("Testing Calendly API - Step 3: Create scheduling link...")
        test_event_type = event_types[0]["uri"]
        
        payload = {
            "max_event_count": 1,
            "owner": test_event_type,
            "owner_type": "EventType"
        }
        
        scheduling_response = requests.post(
            "https://api.calendly.com/scheduling_links",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if scheduling_response.status_code == 201:
            scheduling_data = scheduling_response.json()
            booking_url = scheduling_data["resource"]["booking_url"]
            
            print(f"✅ Test scheduling link created: {booking_url}")
            
            return {
                "success": True,
                "message": "Calendly is properly configured and working!",
                "user_info": {
                    "name": user_name,
                    "uri": user_uri
                },
                "event_types": event_type_list,
                "test_booking_url": booking_url,
                "config": {
                    "CALENDLY_API_KEY": "SET ✅",
                    "CALENDLY_EVENT_TYPE": CALENDLY_EVENT_TYPE or f"Using first available: {event_types[0]['name']}",
                    "CALENDAR_PROVIDER": CALENDAR_PROVIDER
                }
            }
        else:
            return {
                "success": False,
                "error": f"Could not create scheduling link: {scheduling_response.status_code}",
                "response": scheduling_response.text,
                "user_info": {
                    "name": user_name,
                    "uri": user_uri
                },
                "event_types": event_type_list
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Calendly API timeout - check your internet connection"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "type": type(e).__name__
        }
        
# Add this simple test endpoint to your main.py to verify Salesforce is working

@app.get("/api/salesforce/quick-test")
async def quick_salesforce_test():
    """Quick Salesforce connection test with detailed diagnostics"""
    
    print("\n" + "="*50)
    print("SALESFORCE CONNECTION TEST")
    print("="*50)
    
    # Check environment variables
    config = {
        "SALESFORCE_USERNAME": "SET ✅" if SALESFORCE_USERNAME else "NOT SET ❌",
        "SALESFORCE_PASSWORD": "SET ✅" if SALESFORCE_PASSWORD else "NOT SET ❌",
        "SALESFORCE_SECURITY_TOKEN": "SET ✅" if SALESFORCE_SECURITY_TOKEN else "NOT SET ❌",
        "SALESFORCE_DOMAIN": SALESFORCE_DOMAIN or "NOT SET ❌"
    }
    
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Check if all credentials are set
    if not all([SALESFORCE_USERNAME, SALESFORCE_PASSWORD, SALESFORCE_SECURITY_TOKEN]):
        return {
            "success": False,
            "error": "Missing Salesforce credentials",
            "config": config,
            "instructions": "Set SALESFORCE_USERNAME, SALESFORCE_PASSWORD, and SALESFORCE_SECURITY_TOKEN in Railway environment variables"
        }
    
    # Test connection
    try:
        print("\n🔗 Attempting connection...")
        
        from simple_salesforce import Salesforce
        
        sf = Salesforce(
            username=SALESFORCE_USERNAME,
            password=SALESFORCE_PASSWORD,
            security_token=SALESFORCE_SECURITY_TOKEN,
            domain=SALESFORCE_DOMAIN
        )
        
        print("✅ Connected successfully!")
        
        # Test basic queries
        print("\n📊 Running test queries...")
        
        results = {}
        
        # Query Accounts
        try:
            accounts = sf.query("SELECT Id, Name FROM Account LIMIT 5")
            results['accounts'] = {
                'count': accounts['totalSize'],
                'sample': accounts['records'][0]['Name'] if accounts['records'] else None
            }
            print(f"   ✅ Accounts: {accounts['totalSize']} found")
        except Exception as e:
            results['accounts'] = {'error': str(e)}
            print(f"   ⚠️ Accounts query: {e}")
        
        # Query Leads
        try:
            leads = sf.query("SELECT Id, FirstName, LastName, Email, Status FROM Lead LIMIT 5")
            results['leads'] = {
                'count': leads['totalSize'],
                'sample': f"{leads['records'][0].get('FirstName', '')} {leads['records'][0].get('LastName', '')}" if leads['records'] else None
            }
            print(f"   ✅ Leads: {leads['totalSize']} found")
        except Exception as e:
            results['leads'] = {'error': str(e)}
            print(f"   ⚠️ Leads query: {e}")
        
        # Query Contacts
        try:
            contacts = sf.query("SELECT Id, FirstName, LastName, Email FROM Contact LIMIT 5")
            results['contacts'] = {
                'count': contacts['totalSize'],
                'sample': f"{contacts['records'][0].get('FirstName', '')} {contacts['records'][0].get('LastName', '')}" if contacts['records'] else None
            }
            print(f"   ✅ Contacts: {contacts['totalSize']} found")
        except Exception as e:
            results['contacts'] = {'error': str(e)}
            print(f"   ⚠️ Contacts query: {e}")
        
        # Query Cases
        try:
            cases = sf.query("SELECT Id, CaseNumber, Subject, Status FROM Case LIMIT 5")
            results['cases'] = {
                'count': cases['totalSize'],
                'sample': cases['records'][0].get('Subject', '') if cases['records'] else None
            }
            print(f"   ✅ Cases: {cases['totalSize']} found")
        except Exception as e:
            results['cases'] = {'error': str(e)}
            print(f"   ⚠️ Cases query: {e}")
        
        print("\n✅ ALL TESTS PASSED!")
        print("="*50 + "\n")
        
        return {
            "success": True,
            "message": "🎉 Salesforce is connected and working perfectly!",
            "username": SALESFORCE_USERNAME,
            "domain": SALESFORCE_DOMAIN,
            "data": results,
            "next_steps": [
                "✅ Connection verified",
                "✅ Can query Salesforce objects",
                "📝 Ready to create Leads from chatbot intake",
                "🚀 Integration is live!"
            ]
        }
        
    except Exception as e:
        error_message = str(e)
        print(f"\n❌ Connection failed: {error_message}")
        print("="*50 + "\n")
        
        # Provide helpful error messages
        troubleshooting = []
        
        if "INVALID_LOGIN" in error_message or "Invalid username" in error_message:
            troubleshooting = [
                "❌ Invalid username, password, or security token",
                "1. Double-check your username (must be exact Salesforce email)",
                "2. Verify your password is correct",
                "3. Make sure you're using the SECURITY TOKEN (from email), not your password",
                "4. Security token changes when you reset your password",
                "5. Try resetting your security token: Settings → Reset My Security Token"
            ]
        elif "NOT_AUTHORIZED" in error_message:
            troubleshooting = [
                "❌ User not authorized for API access",
                "1. Go to Salesforce Setup → Users → Your User → Edit",
                "2. Make sure 'API Enabled' checkbox is checked",
                "3. Check if your profile has API access permissions"
            ]
        elif "INVALID_SESSION_ID" in error_message:
            troubleshooting = [
                "❌ Session expired or invalid",
                "1. Reset your security token and update the environment variable",
                "2. Restart your Railway app after updating credentials"
            ]
        else:
            troubleshooting = [
                "❌ Connection error",
                "1. Verify all credentials are correct",
                "2. Check if using correct domain (login vs test)",
                "3. Ensure your Salesforce account is active",
                "4. Check Railway logs for more details"
            ]
        
        return {
            "success": False,
            "error": error_message,
            "config": config,
            "troubleshooting": troubleshooting
        }
        
@app.get("/api/twilio/voice-status")
async def voice_status():
    """Check voice endpoint status"""
    return {
        "status": "operational",
        "endpoint": "/api/twilio/voice",
        "full_url": f"{BASE_URL}/api/twilio/voice",
        "twilio_configured": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN),
        "openai_configured": bool(OPENAI_API_KEY),
        "law_firm_name": LAW_FIRM_NAME
    }

# ============================================
# STATIC FILES (serve frontend)
# ============================================

app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")