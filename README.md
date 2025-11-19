# 🏛️ Law Firm AI Chatbot - Complete System

**Enterprise-grade AI chatbot system for law firms with:**
- ✅ OpenAI GPT-4 powered conversations
- ✅ Real-time website content scraping
- ✅ Stripe + PayPal payment processing
- ✅ Automated email notifications
- ✅ Document upload/download
- ✅ Phone integration (Twilio)
- ✅ Client account management
- ✅ Case intake & tracking

---

## 📋 Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Deployment](#deployment)
7. [Usage](#usage)
8. [API Documentation](#api-documentation)
9. [Troubleshooting](#troubleshooting)
10. [Cost Estimates](#cost-estimates)

---

## 🎯 Features

### **Core Functionality**
- **Intelligent Conversations**: OpenAI-powered chatbot with legal knowledge
- **Dynamic Content**: Scrapes your website for up-to-date information
- **Multi-Channel**: Web chat, phone calls, and SMS
- **Client Intake**: Automated case qualification and lead capture
- **Document Management**: Upload/download legal documents securely

### **Integrations**
- **Payments**: Stripe & PayPal for consultation fees, retainers
- **Email**: Automated notifications and follow-ups
- **Phone**: Twilio integration for voice calls and SMS
- **Database**: PostgreSQL for production, SQLite for development

### **Security & Compliance**
- Secure file storage
- Client data encryption
- HIPAA-ready architecture
- Audit logging

---

## 🛠️ Tech Stack

**Backend:**
- FastAPI (Python 3.11+)
- SQLAlchemy (ORM)
- PostgreSQL / SQLite
- OpenAI GPT-4o-mini

**Frontend:**
- Vanilla JavaScript
- HTML5/CSS3
- Responsive design

**Services:**
- OpenAI API
- Stripe
- PayPal
- Twilio
- SMTP (Gmail/SendGrid)

---

## ✅ Prerequisites

1. **Python 3.11+**
2. **API Keys:**
   - OpenAI API key ([get here](https://platform.openai.com/api-keys))
   - Stripe account ([get here](https://dashboard.stripe.com/test/apikeys))
   - PayPal developer account ([get here](https://developer.paypal.com))
   - Twilio account ([get here](https://console.twilio.com))
   - SMTP credentials (Gmail or SendGrid)

3. **Optional:**
   - Railway account for deployment
   - Domain name for production

---

## 📦 Installation

### **1. Clone Repository**

```bash
# Create project directory
mkdir law-firm-chatbot
cd law-firm-chatbot
```

### **2. Set Up Python Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set Up Environment Variables**

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your actual API keys
nano .env  # or use your preferred editor
```

### **5. Initialize Database**

```bash
# Database will be created automatically on first run
# For development, SQLite is used by default
python main.py
```

---

## ⚙️ Configuration

### **1. OpenAI Configuration**

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini  # Cheaper, or use gpt-4o
MAX_TOKENS=500
TEMPERATURE=0.7
```

### **2. Payment Configuration**

**Stripe:**
```env
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
```

**PayPal:**
```env
PAYPAL_CLIENT_ID=your-client-id
PAYPAL_SECRET=your-secret
PAYPAL_MODE=sandbox  # or 'live'
```

### **3. Twilio Configuration**

```env
TWILIO_ACCOUNT_SID=ACxxxxx
TWILIO_AUTH_TOKEN=your-token
TWILIO_PHONE_NUMBER=+1234567890
```

### **4. Email Configuration**

**Gmail:**
```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

**SendGrid (Recommended for production):**
```env
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASSWORD=your-sendgrid-api-key
```

### **5. Law Firm Details**

```env
LAW_FIRM_NAME=Smith & Associates
LAW_FIRM_EMAIL=info@smithlaw.com
LAW_FIRM_PHONE=+1-555-123-4567
LAW_FIRM_WEBSITE=https://smithlaw.com
```

---

## 🚀 Deployment

### **Option 1: Railway (Recommended)**

1. **Install Railway CLI:**
```bash
npm install -g @railway/cli
```

2. **Login to Railway:**
```bash
railway login
```

3. **Initialize Project:**
```bash
railway init
```

4. **Add PostgreSQL:**
```bash
railway add postgresql
```

5. **Set Environment Variables:**
```bash
# Add all variables from .env
railway variables set OPENAI_API_KEY=sk-your-key
railway variables set STRIPE_SECRET_KEY=sk_test_xxx
# ... add all others
```

6. **Deploy:**
```bash
railway up
```

7. **Get URL:**
```bash
railway domain
```

### **Option 2: Manual Deployment (VPS)**

1. **Set up server** (Ubuntu 22.04+)

2. **Install dependencies:**
```bash
sudo apt update
sudo apt install python3.11 python3-pip postgresql nginx
```

3. **Clone and setup:**
```bash
git clone <your-repo>
cd law-firm-chatbot
pip install -r requirements.txt
```

4. **Configure PostgreSQL:**
```bash
sudo -u postgres createdb law_firm_db
```

5. **Set up Nginx:**
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

6. **Run with systemd:**
```bash
sudo nano /etc/systemd/system/lawfirm-bot.service
```

```ini
[Unit]
Description=Law Firm Chatbot
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/law-firm-chatbot
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable lawfirm-bot
sudo systemctl start lawfirm-bot
```

---

## 💻 Usage

### **Local Development**

```bash
# Start server
uvicorn main:app --reload --port 8000

# API will be available at:
# http://localhost:8000

# API docs:
# http://localhost:8000/docs
```

### **Add Chat Widget to Website**

1. **Upload `chat_widget.html` to your server**

2. **Update API URL in widget:**
```javascript
const API_URL = 'https://your-api-url.com/api';
```

3. **Add to your website:**
```html
<!-- In your website's <head> or before </body> -->
<script src="/path/to/chat_widget.html" defer></script>
```

Or **embed inline:**
```html
<!-- Copy entire chat_widget.html content into your page -->
```

### **Configure Twilio Webhooks**

1. Go to Twilio Console → Phone Numbers
2. Set Voice webhook: `https://your-api.com/api/twilio/voice`
3. Set SMS webhook: `https://your-api.com/api/twilio/sms`

---

## 📚 API Documentation

### **Chat Endpoint**

```http
POST /api/chat
Content-Type: application/json

{
  "message": "I need help with a car accident case",
  "session_id": "session_12345",  // optional
  "client_id": "client_67890"     // optional
}
```

**Response:**
```json
{
  "response": "I'm sorry to hear about your accident. Let me ask you a few questions...",
  "session_id": "session_12345",
  "timestamp": "2025-11-18T10:30:00Z"
}
```

### **Create Client**

```http
POST /api/clients
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "+1234567890"
}
```

### **Upload Document**

```http
POST /api/upload
Content-Type: multipart/form-data

file: <file>
case_id: case_12345
client_id: client_67890
```

### **Create Payment**

```http
POST /api/payments/stripe
Content-Type: application/json

{
  "case_id": "case_12345",
  "amount": 250.00
}
```

**Full API documentation:** http://your-api.com/docs

---

## 🐛 Troubleshooting

### **Issue: OpenAI API Error 401**

**Solution:** Check your API key
```bash
# Verify key is set correctly
echo $OPENAI_API_KEY
```

### **Issue: Database Connection Error**

**Solution:** Check DATABASE_URL
```bash
# For development (SQLite)
DATABASE_URL=sqlite:///./law_firm.db

# For production (PostgreSQL)
DATABASE_URL=postgresql://user:password@host:5432/database
```

### **Issue: CORS Errors**

**Solution:** Update allowed origins
```python
# In main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Add your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### **Issue: Files Not Uploading**

**Solution:** Check upload directory permissions
```bash
chmod 755 uploads/
```

### **Issue: Emails Not Sending**

**Solution:** 
1. For Gmail, use App Password: https://support.google.com/accounts/answer/185833
2. Enable "Less secure app access" or use OAuth2

---

## 💰 Cost Estimates

### **Monthly Costs (1000 conversations)**

| Service | Usage | Cost |
|---------|-------|------|
| **OpenAI GPT-4o-mini** | ~700k tokens | $0.15 |
| **Stripe** | 20 payments @ $100 | $60.00 |
| **Twilio** | 50 calls, 200 SMS | $15.00 |
| **SendGrid** | 5000 emails | Free tier |
| **Railway** | Hobby plan | $5.00 |
| **Total** | | **~$80.15/mo** |

**Notes:**
- OpenAI is extremely cheap with GPT-4o-mini
- Most cost is payment processing (can be passed to clients)
- Scale costs linearly with usage

---

## 🔒 Security Considerations

1. **Never commit `.env` to Git**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables for all secrets**

3. **Enable HTTPS in production** (Railway handles this automatically)

4. **Implement rate limiting:**
   ```python
   # Already included in main.py
   RATE_LIMIT_PER_MINUTE=10
   ```

5. **Sanitize user inputs** (handled by Pydantic models)

6. **Regular security updates:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

---

## 📞 Support

**Questions?** 
- Email: support@yourlawfirm.com
- Phone: +1-555-123-4567

**Issues?**
- Check logs: `railway logs` or `journalctl -u lawfirm-bot`
- Review API docs: http://your-api.com/docs

---

## 📄 License

Proprietary - For internal use only

---

## 🎉 Next Steps

1. ✅ Set up all API keys
2. ✅ Test locally
3. ✅ Deploy to Railway
4. ✅ Configure domain
5. ✅ Add chat widget to website
6. ✅ Set up Twilio webhooks
7. ✅ Test all integrations
8. ✅ Go live!

---

**Built with ❤️ for modern law firms**#   l a w - f i r m - b o t  
 