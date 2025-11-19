"""
LAW FIRM AI CHATBOT - BACKEND API
FastAPI backend with OpenAI, Stripe, PayPal, Twilio, Email, File Management
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
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

# ============================================
# WEB SCRAPING SERVICE
# ============================================

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
    """Handles OpenAI conversations with dynamic knowledge"""
    
    def __init__(self):
            
        self.flow_manager = FlowStateManager(self.flow)

        # Build a dictionary of step_id → step
        self.step_index = {}
        for flow in self.flow["flows"]:
            for step in flow["steps"]:
                self.step_index[step["id"]] = step

        self.system_prompt = f"""
        You are an AI legal intake assistant for a law firm. 
        Your behavior is controlled by the following structured flow:

        {json.dumps(self.flow, indent=2)}

        CRITICAL RULES:

        1. Follow the flow EXACTLY as shown above.
        2. Ask ONLY the prompt for the CURRENT step.
        3. Do NOT skip ahead unless the user answers a step fully.
        4. If the user goes off-topic:
        - First answer their question *accurately* using ONLY website content.
        - If the website does not contain the answer: say:
            "I don’t have that information in my records. Would you like me to connect you with a representative?"
        - Then IMMEDIATELY re-ask the pending step’s prompt EXACTLY as written in the JSON.
        5. NEVER invent or approximate legal information or prices.
        6. NEVER generalize (e.g., “usually,” “typically,” “average fee”). 
        7. All firm-specific facts MUST come ONLY from:
        - the provided website content, or
        - the JSON flow script, or
        - the user's previous messages.

        Your job:
        - Identify the current step.
        - Determine whether the user's message answers that step.
        - If yes → move to the next step.
        - If no → answer the side question accurately and then re-ask the current step.

        ALWAYS respond in less than 40 words.
        """


    def get_current_step(self, history):
        """
        Determine which intake step the bot is currently waiting for.
        Looks at the last bot message and matches it to step prompts.
        """
        for msg in reversed(history):
            if msg["role"] == "assistant":
                bot_text = msg["content"].strip().lower()
                for step_id, step in self.step_index.items():
                    if step["prompt"].split("\n")[0].lower() in bot_text:
                        return step_id
                break

        # Default start if nothing matches
        return "start"
    
    def get_next_step(self, current_step_id, user_message):
        step = self.step_index[current_step_id]

        # Handle options (choice / yes_no)
        if "options" in step:
            for opt in step["options"]:
                if opt["label"].lower() in user_message.lower() or opt["value"].lower() in user_message.lower():
                    return opt["next_step"]

        # Text fields have fixed next_step
        if "next_step" in step:
            return step["next_step"]

        return None

    def _load_script(self, script_path: str) -> dict:
        """Load the law_firm.json flow script from disk."""
        try:
            if not os.path.exists(script_path):
                print(f"[WARN] Script file {script_path} not found. Running without script.")
                return {}

            with open(script_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load script {script_path}: {e}")
            return {}

    def _format_flow(self, script: dict) -> str:
        """
        Turn the JSON script into a human-readable list for the model.
        Assumptions:
          law_firm.json has either:
            { "steps": [ { "id": "...", "question": "..." }, ... ] }
          or is simply a list of steps itself.
        """
        if not script:
            return "No scripted flow loaded. You must still perform a logical intake sequence."

        steps = script.get("steps")
        if not steps and isinstance(script, list):
            steps = script

        if not steps:
            return "No 'steps' key found in law_firm.json. Use your best judgment for intake."

        lines = []
        for idx, step in enumerate(steps, start=1):
            step_id = step.get("id", f"step_{idx}")
            question = step.get("question") or step.get("prompt") or ""
            lines.append(f"{idx}. [{step_id}] QUESTION: {question}")

        return "\n".join(lines)

    async def chat(self, message: str, conversation_history: List[Dict], knowledge_base: str = "", current_step_id: str | None = None, current_step_prompt: str | None = None,) -> str:
        """Generate chatbot response using OpenAI"""

        if not openai_client:
            return "I apologize, but the AI service is not configured. Please contact our office directly."

        # Build conversation summary for context (your existing logic)
        summary = self._build_conversation_summary(conversation_history)

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add conversation summary
        if summary:
            messages.append({
                "role": "system",
                "content": (
                    "📊 INFORMATION COLLECTED SO FAR (from user messages):\n"
                    f"{summary}\n\n"
                    "⚠️ Do NOT ask again about anything listed above. "
                    "Use this to decide which script step is currently active."
                )
            })

        # Add knowledge base if available
        if knowledge_base:
            messages.append({
                "role": "system",
                "content": f"Law Firm Website Content:\n\n{knowledge_base}"
            })
            
        messages.insert(1, {
            "role": "system",
            "content": f"LAW FIRM INTAKE FLOW STRUCTURE:\n\n{json.dumps(self.flow, indent=2)}"
        })
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        if current_step_id and current_step_prompt:
            messages.append({
                "role": "system",
                "content": (
                    f"CURRENT INTAKE STEP ID: {current_step_id}\n"
                    f"CURRENT INTAKE PROMPT (must be the question you ask next):\n"
                    f"{current_step_prompt}\n\n"
                    "Do NOT re-ask any earlier questions. Do NOT change this prompt. "
                    "If the user goes off-topic, answer briefly and then repeat THIS prompt."
                )
            })


        # Add recent conversation history (unchanged)
        messages.extend(conversation_history[-10:] if len(conversation_history) > 10 else conversation_history)

        # Add current message
        messages.append({"role": "user", "content": message})

        try:
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please call our office or email us directly."

    def _build_conversation_summary(self, conversation_history: List[Dict]) -> str:
        # (keep your existing implementation here)
        if not conversation_history:
            return ""

        collected = []
        full_text = " ".join(
            [msg.get("content", "") for msg in conversation_history if msg.get("role") == "user"]
        ).lower()
        
        # Detect case type
        if any(keyword in full_text for keyword in ["accident", "crash", "hit", "injured", "hurt", "collision", "rear-end", "slip", "fall", "medical malpractice"]):
            collected.append("✅ Case type: PERSONAL INJURY")
        elif any(keyword in full_text for keyword in ["divorce", "custody", "child support", "separation", "alimony", "visitation"]):
            collected.append("✅ Case type: FAMILY LAW")
        elif any(keyword in full_text for keyword in ["visa", "green card", "citizenship", "immigration", "deportation", "asylum"]):
            collected.append("✅ Case type: IMMIGRATION")
        elif any(keyword in full_text for keyword in ["arrested", "charged", "dui", "dwi", "criminal", "police"]):
            collected.append("✅ Case type: CRIMINAL DEFENSE")
        elif any(keyword in full_text for keyword in ["contract", "business", "partnership", "llc", "corporation"]):
            collected.append("✅ Case type: BUSINESS LAW")
        elif any(keyword in full_text for keyword in ["will", "estate", "trust", "inheritance", "probate"]):
            collected.append("✅ Case type: ESTATE PLANNING")
        
        # Check for incident description
        if len(full_text.split()) > 10 and any(keyword in full_text for keyword in ["happened", "accident", "incident", "was", "were"]):
            collected.append("✅ Incident described")
        
        # Check for date/timing
        if any(keyword in full_text for keyword in ["yesterday", "today", "last week", "last month", "ago", "/202", "/2025", "november", "october", "january"]):
            collected.append("✅ Date/timing mentioned")
        
        # Check for medical treatment
        if any(keyword in full_text for keyword in ["hospital", "doctor", "treatment", "medical", "emergency room", "er", "ambulance", "clinic", "physician"]):
            collected.append("✅ Medical treatment discussed")
        
        # Check for attorney status
        if any(keyword in full_text for keyword in ["attorney", "lawyer", "no attorney", "don't have", "haven't hired"]):
            collected.append("✅ Attorney status mentioned")
        
        # Check for contact info (phone)
        if any(char.isdigit() for char in full_text) and len([c for c in full_text if c.isdigit()]) >= 10:
            collected.append("✅ Phone number provided")
        
        # Check for email
        if "@" in full_text and "." in full_text:
            collected.append("✅ Email provided")
        
        # Check for name
        user_messages = [msg.get("content", "") for msg in conversation_history if msg.get("role") == "user"]
        for msg in user_messages:
            words = msg.split()
            # Look for capitalized words that might be names (not at start of sentence)
            for i, word in enumerate(words):
                if i > 0 and word and word[0].isupper() and len(word) > 2:
                    collected.append("✅ Name mentioned")
                    break
        
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
    """Main chat endpoint"""
    
    # Get or create session
    session_id = chat.session_id or str(uuid.uuid4())
    
    # Get conversation history
    conversation = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).first()
    
    if not conversation:
        conversation = Conversation(
            session_id=session_id,
            client_id=chat.client_id,
            messages=[],
            channel="web"
        )
        db.add(conversation)
    
    # Get knowledge base from website
    knowledge_base = await scraper.get_knowledge_base()
    
    # --- NEW: flow state update based on this user turn ---
    flow_manager.ensure_session(session_id)
    # treat the incoming user message as answering the *previous* step
    current_before = flow_manager.get_current_step(session_id)
    current_after = flow_manager.advance_if_answered(session_id, chat.message)
    current_step_id = current_after
    current_step_prompt = flow_manager.get_prompt(current_step_id)

    # Generate response with awareness of the current step
    response_text = await chatbot.chat(
        message=chat.message,
        conversation_history=conversation.messages,
        knowledge_base=knowledge_base,
        current_step_id=current_step_id,
        current_step_prompt=current_step_prompt,
    )
    
    # Generate response
    response_text = await chatbot.chat(
        message=chat.message,
        conversation_history=conversation.messages,
        knowledge_base=knowledge_base
    )
    
    # Update conversation
    conversation.messages.append({"role": "user", "content": chat.message})
    conversation.messages.append({"role": "assistant", "content": response_text})
    conversation.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    
    return {
        "response": response_text,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
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
    case_id: str = Form(...),
    client_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """Upload document"""
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    new_filename = f"{file_id}{file_ext}"
    file_path = UPLOAD_DIR / new_filename
    
    # Save file
    with open(file_path, "wb") as f:
        content = await file.read()
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
# HEALTH CHECK
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
            "email": bool(SMTP_USER)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)