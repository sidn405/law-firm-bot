# 🚀 Complete Deployment Guide - Law Firm AI Chatbot

## 📦 What You've Got

This is a **complete, production-ready** law firm chatbot system with:

### ✅ **Core Features Implemented**
- ✅ OpenAI GPT-4o-mini powered conversations
- ✅ Real-time website content scraping
- ✅ Structured intake flows (from law_firm.json)
- ✅ Stripe + PayPal payment processing
- ✅ Email notifications (SMTP)
- ✅ Document upload/download
- ✅ Phone integration via Twilio (voice + SMS)
- ✅ Client account lookup and management
- ✅ PostgreSQL database (or SQLite for dev)
- ✅ Professional web chat widget
- ✅ RESTful API with automatic documentation

---

## 📁 File Structure

```
law-firm-chatbot/
├── main.py                    # Main FastAPI backend
├── flow_integration.py        # JSON flow management system
├── chat_widget.html           # Frontend chat widget
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── setup.sh                  # Quick setup script
├── railway.toml              # Railway deployment config
├── README.md                 # Full documentation
└── law_firm.json             # Your intake flows (add this!)
```

---

## ⚡ Quick Start (3 Options)

### **Option 1: Local Development (Fastest)**

```bash
# 1. Run setup script
chmod +x setup.sh
./setup.sh

# 2. Edit .env with your API keys
nano .env

# 3. Add your law_firm.json file
# (Copy the file you uploaded earlier)

# 4. Start server
python3 main.py

# Access at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### **Option 2: Railway (Production - Recommended)**

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Create new project
railway init

# 4. Add PostgreSQL database
railway add postgresql

# 5. Set environment variables (from .env.example)
railway variables set OPENAI_API_KEY=sk-your-key
railway variables set STRIPE_SECRET_KEY=sk_test_xxx
railway variables set TWILIO_ACCOUNT_SID=ACxxx
# ... add all required variables

# 6. Deploy
railway up

# 7. Get your URL
railway domain

# 8. Add your law_firm.json via Railway dashboard
```

### **Option 3: Manual VPS (Advanced)**

See README.md for detailed VPS setup instructions.

---

## 🔑 Required API Keys

### **1. OpenAI (Required)**
- Go to: https://platform.openai.com/api-keys
- Create new API key
- Add to `.env`: `OPENAI_API_KEY=sk-...`
- **Cost**: ~$0.15 per 1000 conversations

### **2. Stripe (Optional - for payments)**
- Go to: https://dashboard.stripe.com/test/apikeys
- Get secret key and publishable key
- Add to `.env`:
  ```
  STRIPE_SECRET_KEY=sk_test_...
  STRIPE_PUBLISHABLE_KEY=pk_test_...
  ```
- **Cost**: 2.9% + $0.30 per transaction

### **3. PayPal (Optional - for payments)**
- Go to: https://developer.paypal.com/developer/applications
- Create app and get credentials
- Add to `.env`:
  ```
  PAYPAL_CLIENT_ID=your-client-id
  PAYPAL_SECRET=your-secret
  ```

### **4. Twilio (Optional - for phone/SMS)**
- Go to: https://console.twilio.com
- Get Account SID, Auth Token, and phone number
- Add to `.env`:
  ```
  TWILIO_ACCOUNT_SID=ACxxx
  TWILIO_AUTH_TOKEN=xxx
  TWILIO_PHONE_NUMBER=+1234567890
  ```
- **Cost**: ~$0.10/min calls, $0.0075/SMS

### **5. Email SMTP (Optional)**

**Gmail (Free):**
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=app-password  # Not regular password!
```
Get app password: https://support.google.com/accounts/answer/185833

**SendGrid (Better for production):**
```
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASSWORD=your-sendgrid-api-key
```
- **Cost**: Free for 100 emails/day

---

## 🌐 Website Integration

### **Step 1: Upload Chat Widget**

1. Upload `chat_widget.html` to your website
2. Update API URL in the file:
   ```javascript
   const API_URL = 'https://your-railway-app.up.railway.app/api';
   ```

### **Step 2: Add to Website**

**Option A: Include File**
```html
<script src="/path/to/chat_widget.html" defer></script>
```

**Option B: Embed Inline**
```html
<!-- Copy entire chat_widget.html content here -->
<script>
  // Chat widget code here
</script>
```

### **Step 3: Test**
1. Visit your website
2. Look for purple chat bubble in bottom-right
3. Click to open chat
4. Test a conversation

---

## 📞 Phone Integration (Twilio)

### **Setup Voice Calls**

1. Go to Twilio Console → Phone Numbers
2. Select your number
3. Under "Voice & Fax":
   - **A CALL COMES IN**: `https://your-app.up.railway.app/api/twilio/voice`
   - Method: `POST`

### **Setup SMS**

1. Same phone number settings
2. Under "Messaging":
   - **A MESSAGE COMES IN**: `https://your-app.up.railway.app/api/twilio/sms`
   - Method: `POST`

### **Test**

1. Call your Twilio number
2. You should hear: "Thank you for calling our law firm..."
3. Speak your question
4. AI will respond via voice

---

## 📊 Flow System (law_firm.json)

Your `law_firm.json` file defines structured intake flows. The system automatically:

1. **Detects keywords** in conversation
   - "personal injury" → starts personal injury intake
   - "family law" → starts family law intake
   - etc.

2. **Follows structured steps** from JSON
   - Asks predefined questions
   - Validates responses
   - Collects data in order

3. **Saves to database** when complete
   - Creates client record
   - Creates case with intake data
   - Sends email to law firm

### **How It Works**

User says: "I was in a car accident"

→ System detects "car accident" keyword
→ Starts `personal_injury_intake` flow
→ Asks questions from JSON file
→ Saves responses to database
→ Emails law firm when complete

**Hybrid Mode**: System uses OpenAI for general conversation but switches to structured flows for intake.

---

## 🧪 Testing Checklist

### **Local Testing**

- [ ] API starts without errors
- [ ] Can access http://localhost:8000/docs
- [ ] Health check passes: http://localhost:8000/health
- [ ] Chat widget loads on test page
- [ ] Can send messages and get responses
- [ ] File upload works
- [ ] Database creates records

### **Integration Testing**

- [ ] OpenAI responses are relevant
- [ ] Website content is being scraped
- [ ] Structured flows work (test with "personal injury")
- [ ] Email notifications send
- [ ] Stripe payment test works
- [ ] Twilio webhooks respond

### **Production Testing**

- [ ] HTTPS enabled
- [ ] Environment variables set
- [ ] Database connected
- [ ] API accessible from website
- [ ] Chat widget shows on site
- [ ] Phone calls work
- [ ] SMS works
- [ ] All features functional

---

## 💰 Cost Breakdown

### **Monthly Costs (Estimated for 1000 conversations)**

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| OpenAI GPT-4o-mini | ~700k tokens | $0.15 |
| Railway Hosting | Hobby plan | $5.00 |
| Twilio (Phone) | 50 calls @ 5min | $25.00 |
| Twilio (SMS) | 200 messages | $1.50 |
| SendGrid Email | 5000 emails | Free |
| Stripe Fees | 20 × $100 payments | $60.00 |
| **Total** | | **$91.65/mo** |

**Notes:**
- OpenAI is very cheap with GPT-4o-mini
- Most cost is payment processing (pass to clients)
- Twilio cost depends on actual usage
- Can start with free tiers for testing

---

## 🔒 Security Best Practices

### **Before Going Live**

1. **Change all API keys to production**
   ```bash
   # Use production keys, not test keys
   OPENAI_API_KEY=sk-prod-...
   STRIPE_SECRET_KEY=sk_live_...
   PAYPAL_MODE=live
   ```

2. **Enable HTTPS** (Railway does this automatically)

3. **Set strong SECRET_KEY**
   ```bash
   openssl rand -hex 32
   ```

4. **Restrict CORS origins**
   ```python
   # In main.py, update:
   allow_origins=["https://yourlawfirm.com"]
   ```

5. **Set up rate limiting** (already included)

6. **Regular backups** of database

7. **Monitor logs**
   ```bash
   railway logs
   ```

---

## 🐛 Common Issues & Solutions

### **Issue: "OpenAI API Error 401"**
**Fix:** Check API key is correct and has billing enabled

### **Issue: "CORS Error"**
**Fix:** Add your domain to `allow_origins` in main.py

### **Issue: "Database Error"**
**Fix:** Check DATABASE_URL is correct

### **Issue: "Twilio Webhooks Not Working"**
**Fix:** 
1. Make sure URLs use HTTPS
2. Check Twilio webhook configuration
3. Review Twilio error logs

### **Issue: "Emails Not Sending"**
**Fix:**
1. For Gmail, use App Password, not regular password
2. Enable "Less secure app access"
3. Check SMTP credentials

### **Issue: "Files Not Uploading"**
**Fix:**
1. Check `uploads/` directory exists
2. Verify file permissions: `chmod 755 uploads/`
3. Check `MAX_FILE_SIZE` setting

---

## 📈 Scaling Considerations

### **When You Grow**

**Up to 1,000 conversations/month:**
- Railway Hobby plan: $5/mo
- SQLite database: Free
- Current setup works fine

**Up to 10,000 conversations/month:**
- Upgrade to Railway Pro: $20/mo
- Switch to PostgreSQL
- Add Redis for caching
- Consider SendGrid for emails

**Up to 100,000+ conversations/month:**
- Dedicated server or AWS/GCP
- PostgreSQL with replication
- Load balancer
- CDN for chat widget
- Monitoring (Sentry, DataDog)

---

## 🎯 Next Steps

### **Day 1: Setup & Testing**
1. [ ] Get all API keys
2. [ ] Configure .env
3. [ ] Test locally
4. [ ] Verify all features work

### **Day 2: Deploy**
1. [ ] Deploy to Railway
2. [ ] Configure database
3. [ ] Set environment variables
4. [ ] Test production deployment

### **Day 3: Integration**
1. [ ] Add chat widget to website
2. [ ] Configure Twilio webhooks
3. [ ] Test all channels (web, phone, SMS)
4. [ ] Train team on system

### **Day 4: Go Live**
1. [ ] Switch to production API keys
2. [ ] Final testing
3. [ ] Monitor for issues
4. [ ] Celebrate! 🎉

---

## 📞 Support

**Technical Issues:**
- Check API logs: `railway logs`
- Review FastAPI docs: http://your-api.com/docs
- Check this README

**Questions:**
- Email: [your-email]
- Documentation: README.md
- API Reference: /docs endpoint

---

## ✅ Production Checklist

Before going live:

- [ ] All API keys set to production
- [ ] HTTPS enabled
- [ ] CORS configured for your domain
- [ ] Database backups enabled
- [ ] Error monitoring set up
- [ ] Email notifications tested
- [ ] Payment processing tested
- [ ] Phone integration tested
- [ ] Chat widget tested on actual website
- [ ] Team trained on system
- [ ] Client data handling verified
- [ ] Privacy policy updated
- [ ] Terms of service reviewed

---

**Built for modern law firms | OpenAI + FastAPI + Railway**

**Questions? Need help?** Check the README.md or deployment logs!