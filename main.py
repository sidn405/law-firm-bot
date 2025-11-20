"""
LAW FIRM AI CHATBOT - BACKEND API
FastAPI backend with OpenAI, Stripe, PayPal, Twilio, Email, File Management
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
# from flow_integration import HybridChatbotService

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
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
LAW_FIRM_EMAIL = os.getenv("LAW_FIRM_EMAIL", "info@lawfirm.com")
LAW_FIRM_PHONE = os.getenv("LAW_FIRM_PHONE", "+1234567890")
CALENDLY_API_KEY = os.getenv("CALENDLY_API_KEY", "")
CALENDLY_EVENT_TYPE = os.getenv("CALENDLY_EVENT_TYPE", "")
GOOGLE_CALENDAR_CREDENTIALS = os.getenv("GOOGLE_CALENDAR_CREDENTIALS", "")
CALENDAR_PROVIDER = os.getenv("CALENDAR_PROVIDER", "calendly")

# Initialize services
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
stripe.api_key = STRIPE_SECRET_KEY
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None



# File storage
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load law firm flow
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLOW_PATH = os.path.join(BASE_DIR, "law_firm", "law_firm.json")  # or just "law_firm.json" if it’s in root

with open(FLOW_PATH, "r", encoding="utf-8") as f:
    LAW_FIRM_FLOW = json.load(f)

flow_manager = FlowStateManager(LAW_FIRM_FLOW)


# ============================================
# DATABASE SETUP
# ============================================

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./law_firm.db")
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
    status = Column(String, default="new")  # new, contacted, active, closed
    
class Case(Base):
    __tablename__ = "cases"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, nullable=False)
    case_type = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default="pending")  # pending, reviewing, accepted, rejected
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
    channel = Column(String, default="web")  # web, phone, sms
    
class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, default="pending")  # pending, completed, failed, refunded
    provider = Column(String)  # stripe, paypal
    transaction_id = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    payment_metadata = Column(JSON)  # Renamed from 'metadata' to avoid SQLAlchemy conflict

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

Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
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

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(title="Law Firm AI Chatbot API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your domain
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
    provider: str  # stripe or paypal
    
class EmailRequest(BaseModel):
    to: str
    subject: str
    body: str
    html: Optional[str] = None
    
class AppointmentRequest(BaseModel):
    client_name: str
    client_email: EmailStr
    client_phone: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    case_type: Optional[str] = None
    notes: Optional[str] = None

# ============================================
# WEB SCRAPING SERVICE
# ============================================

class CalendarService:
    """Manages calendar integrations for appointment scheduling"""
    
    @staticmethod
    async def create_calendly_invitation(appointment_data: dict) -> Dict:
        """Create Calendly scheduling link"""
        if not CALENDLY_API_KEY or not CALENDLY_EVENT_TYPE:
            return {"success": False, "error": "Calendly not configured"}
        
        try:
            headers = {
                "Authorization": f"Bearer {CALENDLY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            user_response = requests.get(
                "https://api.calendly.com/users/me",
                headers=headers
            )
            user_data = user_response.json()
            
            payload = {
                "max_event_count": 1,
                "owner": user_data["resource"]["uri"],
                "owner_type": "EventType"
            }
            
            response = requests.post(
                "https://api.calendly.com/scheduling_links",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 201:
                data = response.json()
                return {
                    "success": True,
                    "booking_url": data["resource"]["booking_url"],
                    "event_id": None
                }
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"Calendly error: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def get_available_slots(date: str = None) -> Dict:
        """Get available appointment slots"""
        if CALENDAR_PROVIDER == "calendly" and CALENDLY_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {CALENDLY_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                user_response = requests.get(
                    "https://api.calendly.com/users/me",
                    headers=headers
                )
                user_data = user_response.json()
                user_uri = user_data["resource"]["uri"]
                
                params = {"user": user_uri, "count": 10}
                if date:
                    params["min_start_time"] = date
                
                response = requests.get(
                    "https://api.calendly.com/event_type_available_times",
                    headers=headers,
                    params=params
                )
                
                if response.status_code == 200:
                    return {"success": True, "slots": response.json()}
                else:
                    return {"success": False, "error": "Could not fetch availability"}
                    
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Calendar not configured"}

calendar_service = CalendarService()

class WebScraperService:
    """Scrapes law firm website for dynamic content"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    async def scrape_page(self, path: str) -> str:
        """Scrape a specific page"""
        url = f"{self.base_url}{path}"
        cache_key = f"page_{path}"
        
        # Check cache
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
                        
                        # Remove script and style elements
                        for script in soup(["script", "style", "nav", "footer"]):
                            script.decompose()
                        
                        # Get text
                        text = soup.get_text()
                        
                        # Clean up
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = '\n'.join(chunk for chunk in chunks if chunk)
                        
                        # Cache
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

# Initialize scraper (update with actual law firm URL)
scraper = WebScraperService(base_url="https://yourlawfirm.com")

class FlowStateManager:
    """
    Tracks the current flow step for each session and advances using law_firm.json.
    """

    def __init__(self, flow_json: dict):
        self.flow = flow_json
        self.step_index = self._index_steps()
        # in-memory store: session_id -> {"current_step": str, "answers": dict}
        self.sessions: dict[str, dict] = {}

    def _index_steps(self) -> dict:
        index = {}
        for flow in self.flow["flows"]:
            for step in flow["steps"]:
                index[step["id"]] = step
        return index

    # -------- session helpers --------
    def ensure_session(self, session_id: str):
        if session_id not in self.sessions:
            # default entry point for now – matches "start" in main_menu
            self.sessions[session_id] = {
                "current_step": "start",
                "answers": {}
            }

    def get_current_step(self, session_id: str) -> str:
        self.ensure_session(session_id)
        return self.sessions[session_id]["current_step"]

    def set_current_step(self, session_id: str, step_id: str):
        self.ensure_session(session_id)
        self.sessions[session_id]["current_step"] = step_id

    def get_step(self, step_id: str) -> dict:
        return self.step_index[step_id]

    def get_prompt(self, step_id: str) -> str:
        return self.step_index[step_id]["prompt"]

    # -------- answer detection --------
    def did_user_answer_step(self, step_id: str, user_message: str) -> bool:
        """
        Very simple heuristics – enough to stop the 'date' question repeating.
        You can refine per step over time.
        """
        step = self.get_step(step_id)
        msg = (user_message or "").strip().lower()

        itype = step.get("input_type")

        if itype == "none":
            return True

        if itype == "text":
            # any non-empty text counts as an answer
            return len(msg) > 0

        if itype == "choice" or itype == "yes_no":
            # PI intro is special: user may give real date/time, not button values
            if step_id == "pi_intro":
                # crude date/time pattern – catches things like '7:23 am', '11/12/2025', 'nov 12'
                if any(c in msg for c in ["/", "-", ":"]) or any(
                    m in msg for m in
                    ["yesterday", "today", "ago", "last week", "last month",
                     "january", "february", "march", "april", "may", "june",
                     "july", "august", "september", "october", "november", "december"]
                ):
                    return True

            # normal choice / yes_no – match label or value text
            for opt in step.get("options", []):
                if opt["label"].lower() in msg or opt["value"].lower() in msg:
                    return True

        # file upload etc – you can handle later
        return False

    # -------- next step logic --------
    def determine_next_step(self, current_step_id: str, user_message: str) -> str | None:
        step = self.get_step(current_step_id)
        msg = (user_message or "").lower()

        # Personal-injury special case from your JSON:
        # pi_injury_type -> either pi_injury_details (if "other") or pi_medical_treatment
        if current_step_id == "pi_injury_type":
            for opt in step.get("options", []):
                if opt["label"].lower() in msg or opt["value"].lower() in msg:
                    return opt["next_step"]
            # fallback: if they described it in text, treat as pi_injury_details answered
            if len(msg.split()) > 3:
                return "pi_medical_treatment"

        # pi_injury_details should ALWAYS go to pi_medical_treatment
        if current_step_id == "pi_injury_details":
            return step.get("next_step", "pi_medical_treatment")

        # generic path – follow JSON
        if "options" in step:
            for opt in step["options"]:
                if opt["label"].lower() in msg or opt["value"].lower() in msg:
                    return opt.get("next_step")

        return step.get("next_step")

    def advance_if_answered(self, session_id: str, user_message: str):
        """
        If user answered the current step, move to the next one.
        Otherwise leave current_step alone (off-topic or incomplete).
        """
        self.ensure_session(session_id)
        current = self.sessions[session_id]["current_step"]

        if not self.did_user_answer_step(current, user_message):
            return current  # stay on same step

        nxt = self.determine_next_step(current, user_message)
        if nxt:
            self.sessions[session_id]["current_step"] = nxt
        return self.sessions[session_id]["current_step"]


# ============================================
# OPENAI CHATBOT SERVICE
# ============================================
from flow_state_manager import FlowStateManager
class ChatbotService:
    """Handles OpenAI conversations with strict flow management"""
    
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(base_dir, "law_firm", "law_firm.json")

        with open(script_path, "r", encoding="utf-8") as f:
            self.flow = json.load(f)

        self.flow_manager = FlowStateManager(self.flow)

        # Build step index
        self.step_index = {}
        for flow in self.flow["flows"]:
            for step in flow["steps"]:
                self.step_index[step["id"]] = step

        # Stronger system prompt
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

        # Get step details
        step_data = self.step_index.get(current_step_id, {})
        input_type = step_data.get("input_type", "text")
        options = step_data.get("options", [])

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # CRITICAL: Make the current question unmissable
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

        # Add what we've collected so far (to avoid re-asking)
        collected_info = self._extract_collected_info(conversation_history)
        if collected_info:
            messages.append({
                "role": "system",
                "content": f"✅ Already collected:\n{collected_info}\n\n⚠️ DO NOT ask about these again!"
            })

        # Add recent conversation (last 6 messages to save tokens)
        recent = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        messages.extend(recent)

        # Add current user message
        messages.append({"role": "user", "content": message})

        # Call OpenAI
        try:
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,  # Keep responses short
                temperature=0.3   # More focused
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

        # Check for case type
        if "accident" in all_user_text or "car" in all_user_text:
            collected.append("- Case type: Personal Injury (car accident)")
        
        # Check for timing
        if any(word in all_user_text for word in ["today", "yesterday", "last week", "ago", "/", "2025", "2024"]):
            collected.append("- Incident timing mentioned")
        
        # Check for medical
        if any(word in all_user_text for word in ["hospital", "doctor", "medical", "treatment", "er"]):
            collected.append("- Medical treatment discussed")
        
        # Check for attorney status
        if "attorney" in all_user_text or "lawyer" in all_user_text:
            collected.append("- Attorney status mentioned")

        return "\n".join(collected) if collected else ""

chatbot = ChatbotService()

# ============================================
# EMAIL SERVICE
# ============================================

async def send_email(to: str, subject: str, body: str, html: str = None):
    """Send email via SMTP"""
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = SMTP_USER
        msg['To'] = to
        msg['Subject'] = subject
        
        # Plain text
        msg.attach(MIMEText(body, 'plain'))
        
        # HTML if provided
        if html:
            msg.attach(MIMEText(html, 'html'))
        
        # Send
        await asyncio.to_thread(
            lambda: smtplib.SMTP(SMTP_HOST, SMTP_PORT).starttls() or 
                    smtplib.SMTP(SMTP_HOST, SMTP_PORT).login(SMTP_USER, SMTP_PASSWORD) or
                    smtplib.SMTP(SMTP_HOST, SMTP_PORT).send_message(msg)
        )
        
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# ============================================
# PAYMENT SERVICES
# ============================================

async def create_stripe_payment(amount: float, description: str, metadata: dict = None) -> Dict:
    """Create Stripe payment intent"""
    try:
        intent = await asyncio.to_thread(
            stripe.PaymentIntent.create,
            amount=int(amount * 100),  # Convert to cents
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
            "https://api-m.sandbox.paypal.com/v2/checkout/orders",  # Use production URL for live
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
    """Main chat endpoint - PURE script-driven flow (no LLM for now)"""

    # Get or create session
    session_id = chat.session_id or str(uuid.uuid4())

    # Get conversation
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

    # Ensure flow session exists
    flow_manager.ensure_session(session_id)

    # Previous step (before this message)
    prev_step_id = flow_manager.get_current_step(session_id)

    # Treat this message as the answer to the CURRENT step
    new_step_id = flow_manager.advance_if_answered(session_id, chat.message)

    # Current step AFTER processing the answer
    current_step_id = flow_manager.get_current_step(session_id)

    # Get the prompt for the current step (what we should ask next)
    response_text = flow_manager.get_prompt(current_step_id)

    # Update conversation history
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
    
    # Check if exists
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
    
    # Send welcome email
    await send_email(
        to=client.email,
        subject="Thank you for contacting us",
        body=f"Dear {client.name},\n\nThank you for reaching out. We'll review your case and contact you soon.\n\nBest regards,\nThe Legal Team",
        html=f"<p>Dear {client.name},</p><p>Thank you for reaching out. We'll review your case and contact you soon.</p>"
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
    
    # Get cases
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
    
    # Notify law firm
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
        # Save to database
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
        # Validate file size (10MB max)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            return {"success": False, "error": "File too large. Maximum size is 10MB."}
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        new_filename = f"{file_id}{file_ext}"
        file_path = UPLOAD_DIR / new_filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Save to database
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

@app.post("/api/twilio/process-speech")
async def process_speech(SpeechResult: str = Form(...), CallSid: str = Form(...), db: Session = Depends(get_db)):
    """Process speech input from phone"""
    
    # Get knowledge base
    knowledge_base = await scraper.get_knowledge_base()
    
    # Get or create conversation
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
    
    # Generate response
    response_text = await chatbot.chat(
        message=SpeechResult,
        conversation_history=conversation.messages,
        knowledge_base=knowledge_base
    )
    
    # Update conversation
    conversation.messages.append({"role": "user", "content": SpeechResult})
    conversation.messages.append({"role": "assistant", "content": response_text})
    db.commit()
    
    # Create voice response
    twiml = VoiceResponse()
    
    # Speak response
    twiml.say(response_text, voice='alice')
    
    # Continue gathering
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
    
    # Get knowledge base
    knowledge_base = await scraper.get_knowledge_base()
    
    # Get or create conversation
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
    
    # Generate response
    response_text = await chatbot.chat(
        message=Body,
        conversation_history=conversation.messages,
        knowledge_base=knowledge_base
    )
    
    # Update conversation
    conversation.messages.append({"role": "user", "content": Body})
    conversation.messages.append({"role": "assistant", "content": response_text})
    db.commit()
    
    # Send SMS response
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
# CALENDAR SCHEDULER
# ============================================   
    
@app.post("/api/appointments/schedule")
async def schedule_appointment(appointment: AppointmentRequest, db: Session = Depends(get_db)):
    """Schedule a consultation appointment"""
    
    try:
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
        
        calendar_result = None
        if CALENDAR_PROVIDER == "calendly":
            calendar_result = await calendar_service.create_calendly_invitation({
                "name": appointment.client_name,
                "email": appointment.client_email,
                "phone": appointment.client_phone,
                "notes": appointment.notes
            })
        
        new_appointment = Appointment(
            client_id=client.id,
            client_name=appointment.client_name,
            client_email=appointment.client_email,
            client_phone=appointment.client_phone,
            case_type=appointment.case_type,
            notes=appointment.notes,
            status="pending",
            calendar_event_id=calendar_result.get("event_id") if calendar_result else None,
            calendar_link=calendar_result.get("booking_url") if calendar_result else None
        )
        
        db.add(new_appointment)
        db.commit()
        db.refresh(new_appointment)
        
        # Send confirmation email
        if calendar_result and calendar_result.get("success"):
            email_body = f"""Dear {appointment.client_name},

Thank you for choosing our law firm for your {appointment.case_type or 'legal'} consultation.

NEXT STEP: Please click the link below to select your preferred appointment time:
{calendar_result.get('booking_url')}

Your requested time: {appointment.preferred_date or 'Not specified'}

Once you complete your booking, you'll receive:
- Instant calendar confirmation
- Email reminder 24 hours before
- Text message reminder (if you provided your phone)

Questions? Call us at {LAW_FIRM_PHONE} or reply to this email.

Best regards,
The Legal Team"""
            
            await send_email(
                to=appointment.client_email,
                subject="📅 Complete Your Consultation Booking - Action Required",
                body=email_body
            )
            
            return {
                "success": True,
                "appointment_id": new_appointment.id,
                "calendar_link": calendar_result.get("booking_url"),
                "message": "Please use the calendar link to confirm your appointment time"
            }
        else:
            # Fallback: send email to law firm for manual scheduling
            email_body = f"""New Consultation Request - ACTION REQUIRED

Client Details:
- Name: {appointment.client_name}
- Email: {appointment.client_email}
- Phone: {appointment.client_phone or 'Not provided'}
- Case Type: {appointment.case_type or 'Not specified'}
- Requested Time: {appointment.preferred_date or 'Not specified'}

Additional Notes:
{appointment.notes or 'None'}

Appointment ID: {new_appointment.id}

ACTION: Please contact this client within 2 hours to confirm appointment availability."""
            
            await send_email(
                to=LAW_FIRM_EMAIL,
                subject=f"🔔 New Consultation: {appointment.client_name} - {appointment.case_type}",
                body=email_body
            )
            
            # Email to client
            client_email_body = f"""Dear {appointment.client_name},

Thank you for requesting a consultation with our law firm.

We've received your request for: {appointment.preferred_date or 'a consultation'}
Case type: {appointment.case_type or 'General consultation'}

Our scheduling team will review your preferred time and contact you within 2 hours at:
- Phone: {appointment.client_phone or 'Not provided'}
- Email: {appointment.client_email}

If your requested time isn't available, we'll suggest alternative times that work for you.

Need immediate assistance? Call us at {LAW_FIRM_PHONE}

Best regards,
The Legal Team"""
            
            await send_email(
                to=appointment.client_email,
                subject="✓ Consultation Request Received - We'll Confirm Soon",
                body=client_email_body
            )
            
            return {
                "success": True,
                "appointment_id": new_appointment.id,
                "calendar_link": None,
                "message": "Your consultation request has been received. We'll confirm your appointment within 2 hours."
            }
        
    except Exception as e:
        print(f"Appointment scheduling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

# ============================================
# HEALTH CHECK
# ============================================

@app.post("/api/admin/recreate-db")
async def recreate_database():
    """Temporary endpoint to recreate database tables"""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    return {"success": True, "message": "Database tables recreated"}

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
            "email": bool(SMTP_USER)
        }
    }
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")
