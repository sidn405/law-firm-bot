"""
LAW FIRM AI CHATBOT - BACKEND API
FastAPI backend with OpenAI, Stripe, PayPal, Twilio, Resend Email, File Management
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from fastapi.staticfiles import StaticFiles
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

# ============================================
# CONFIGURATION
# ============================================

# API Keys (set as environment variables)
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

# Dependency
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

# ============================================
# FILE UPLOAD/DOWNLOAD
# ============================================

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    case_id: str = Form(default="temp"),
    client_id: str = Form(default="guest"),
    db: Session = Depends(get_db)
):
    """Upload document"""
    
    try:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            return {"success": False, "error": "File too large. Maximum size is 10MB."}
        
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        new_filename = f"{file_id}{file_ext}"
        file_path = UPLOAD_DIR / new_filename
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        document = Document(
            case_id=case_id,
            client_id=client_id,
            filename=file.filename,
            file_path=str(file_path),
            file_type=file.content_type,
            file_size=len(content)
        )
        db.add(document)
        db.commit()
        
        return {
            "success": True,
            "document_id": document.id,
            "filename": file.filename
        }
    except Exception as e:
        print(f"Upload error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/download/{document_id}")
async def download_file(document_id: str, db: Session = Depends(get_db)):
    """Download document"""
    
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return FileResponse(
        path=document.file_path,
        filename=document.filename,
        media_type=document.file_type
    )

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

@app.post("/api/twilio/appointment-confirmation")
async def appointment_confirmation_call(
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle automated appointment confirmation call with full details"""
    form_data = await request.form()
    appointment_id = request.query_params.get("appointment_id")
    
    print(f"📞 Incoming confirmation call for appointment: {appointment_id}")
    
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    
    response = VoiceResponse()
    
    if not appointment:
        print(f"❌ Appointment not found: {appointment_id}")
        response.say("We're sorry, we couldn't find your appointment. Please call our office.", voice='alice')
        return str(response)
    
    print(f"✅ Found appointment for {appointment.client_name}")
    
    gather = Gather(
        num_digits=1,
        action=f'/api/twilio/confirm-appointment?appointment_id={appointment_id}',
        timeout=10
    )
    
    # Build detailed message with appointment info
    case_type = appointment.case_type or 'consultation'
    date_time = appointment.scheduled_date.strftime('%A, %B %d at %I:%M %p') if appointment.scheduled_date else 'your requested time'
    
    # Parse notes to extract key details
    notes = appointment.notes or ""
    injury_type = "not specified"
    incident_date = "not specified"
    
    if "Injury Type:" in notes:
        try:
            injury_type = notes.split("Injury Type:")[1].split("\n")[0].strip()
        except:
            pass
    
    if "Date/Time:" in notes:
        try:
            incident_date = notes.split("Date/Time:")[1].split("\n")[0].strip()
        except:
            pass
    
    message = f"""Hello {appointment.client_name}. This is {LAW_FIRM_NAME} calling to confirm your {case_type} consultation.

We have you scheduled for {date_time}.

Based on your intake, this is regarding a {injury_type} that occurred {incident_date}.

Our attorney will review:
- Your incident details
- Medical treatment received
- Potential case value
- Next steps for your claim

To confirm this appointment, press 1.
To request a different time, press 2.
To speak with someone now, press 3.
To hear these options again, press 9."""
    
    gather.say(message, voice='alice')
    response.append(gather)
    
    # Fallback if no response
    response.say("We didn't receive a response. We'll send you an email instead. Thank you, goodbye.", voice='alice')
    
    return str(response)


@app.post("/api/twilio/confirm-appointment")
async def confirm_appointment_response(
    request: Request,
    db: Session = Depends(get_db)
):
    """Process appointment confirmation response"""
    form_data = await request.form()
    digits = form_data.get("Digits")
    appointment_id = request.query_params.get("appointment_id")
    
    print(f"📞 Received response: {digits} for appointment: {appointment_id}")
    
    appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    response = VoiceResponse()
    
    if not appointment:
        response.say("We couldn't find your appointment. Please call our office.", voice='alice')
        return str(response)
    
    if digits == "1":
        # CONFIRM APPOINTMENT
        print(f"✅ Appointment CONFIRMED by {appointment.client_name}")
        appointment.status = "confirmed"
        db.commit()
        
        date_str = appointment.scheduled_date.strftime('%A, %B %d at %I:%M %p') if appointment.scheduled_date else 'your scheduled time'
        
        response.say(
            f"""Perfect! Your {appointment.case_type or 'consultation'} is confirmed for {date_str}.
            
            You'll receive a confirmation email with:
            - Office address and directions
            - What to bring to your consultation
            - Our attorney's direct contact information
            
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
[Suite Number]

🚗 PARKING:
[Parking instructions]

📋 WHAT TO BRING:
✓ Photo ID (driver's license or state ID)
✓ Any documents related to your case
✓ Medical records or bills (if applicable)
✓ Police report (if applicable)
✓ Insurance information
✓ List of questions you want to ask

👔 WHAT TO EXPECT:
Our attorney will:
- Review your case details
- Assess the strength of your claim
- Explain your legal options
- Discuss potential case value
- Answer all your questions
- Outline next steps if you decide to proceed

📱 NEED TO RESCHEDULE?
Call us at {LAW_FIRM_PHONE}
Or email {LAW_FIRM_EMAIL}

We look forward to helping you with your {appointment.case_type or 'legal matter'}.

Best regards,
{LAW_FIRM_NAME}
{LAW_FIRM_PHONE}""",
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
                        [Office Address Here]<br>
                        [Suite Number]<br>
                        <a href="https://maps.google.com" style="color: #2563eb;">Get Directions</a>
                    </p>
                    
                    <h3 style="color: #1f2937;">📋 What to Bring</h3>
                    <ul style="background-color: white; padding: 20px; border-radius: 6px;">
                        <li>Photo ID (driver's license or state ID)</li>
                        <li>Any documents related to your case</li>
                        <li>Medical records or bills (if applicable)</li>
                        <li>Police report (if applicable)</li>
                        <li>Insurance information</li>
                        <li>List of questions you want to ask</li>
                    </ul>
                    
                    <h3 style="color: #1f2937;">👔 What to Expect</h3>
                    <p style="background-color: white; padding: 15px; border-radius: 6px;">
                        Our attorney will review your case details, assess the strength of your claim, 
                        explain your legal options, and discuss potential case value. This is your opportunity 
                        to ask questions and learn about the legal process.
                    </p>
                    
                    <div style="background-color: #dbeafe; padding: 15px; border-radius: 6px; margin-top: 20px; border-left: 4px solid #2563eb;">
                        <p style="margin: 0;"><strong>📱 Need to reschedule?</strong></p>
                        <p style="margin: 10px 0 0 0;">
                            Call <a href="tel:{LAW_FIRM_PHONE}" style="color: #2563eb;">{LAW_FIRM_PHONE}</a><br>
                            Email <a href="mailto:{LAW_FIRM_EMAIL}" style="color: #2563eb;">{LAW_FIRM_EMAIL}</a>
                        </p>
                    </div>
                </div>
                
                <div style="background-color: #1f2937; color: white; padding: 15px; border-radius: 0 0 8px 8px; text-align: center;">
                    <p style="margin: 0;">We look forward to helping you!</p>
                    <p style="margin: 5px 0;"><strong>{LAW_FIRM_NAME}</strong></p>
                    <p style="margin: 5px 0;">{LAW_FIRM_PHONE}</p>
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
            f"""No problem, {appointment.client_name}. 
            
            We'll have someone from our scheduling team call you within one hour to find a better time that works for you.
            
            If you need immediate assistance, please call our office at {LAW_FIRM_PHONE}.
            
            Thank you, goodbye.""",
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
Original Time: {appointment.scheduled_date.strftime('%B %d, %Y at %I:%M %p') if appointment.scheduled_date else 'Not set'}
Case Type: {appointment.case_type}

⏰ ACTION REQUIRED: Call client within 1 hour to reschedule.

Appointment ID: {appointment_id}""",
            html=f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #dc2626;">🔄 RESCHEDULE REQUEST</h2>
                <div style="background-color: #fee; padding: 15px; border-left: 4px solid #dc2626; margin: 20px 0;">
                    <strong>⏰ ACTION REQUIRED:</strong> Call client within 1 hour to reschedule
                </div>
                <h3>Client Details:</h3>
                <ul>
                    <li><strong>Name:</strong> {appointment.client_name}</li>
                    <li><strong>Phone:</strong> <a href="tel:{appointment.client_phone}">{appointment.client_phone}</a></li>
                    <li><strong>Email:</strong> {appointment.client_email}</li>
                    <li><strong>Case Type:</strong> {appointment.case_type}</li>
                    <li><strong>Original Time:</strong> {appointment.scheduled_date.strftime('%B %d, %Y at %I:%M %p') if appointment.scheduled_date else 'Not set'}</li>
                </ul>
                <p><small>Appointment ID: {appointment_id}</small></p>
            </div>
            """
        )
        
    elif digits == "3":
        # SPEAK WITH SOMEONE NOW
        print(f"📞 Transfer requested by {appointment.client_name}")
        response.say(
            "Transferring you to our office now. Please hold.",
            voice='alice'
        )
        response.dial(LAW_FIRM_PHONE)
        
    elif digits == "9":
        # REPEAT OPTIONS
        return await appointment_confirmation_call(request, db)
        
    else:
        response.say("I didn't understand that option. Goodbye.", voice='alice')
    
    return str(response)

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

# ============================================
# STATIC FILES (serve frontend)
# ============================================

app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")