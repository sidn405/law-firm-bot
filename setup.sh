#!/bin/bash

# ============================================
# LAW FIRM CHATBOT - QUICK START SCRIPT
# ============================================

echo "🏛️  Law Firm AI Chatbot - Quick Start"
echo "======================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Mac/Linux
    source venv/bin/activate
fi

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "✓ .env created - Please edit it with your API keys!"
    echo ""
    echo "⚠️  IMPORTANT: You must edit .env with your actual API keys before running!"
    echo ""
fi

# Create uploads directory
echo "📁 Creating uploads directory..."
mkdir -p uploads
touch uploads/.gitkeep

# Copy law_firm.json if needed
if [ ! -f law_firm.json ]; then
    echo ""
    echo "⚠️  law_firm.json not found - Please add your intake flows!"
fi

# Check if .env is configured
if grep -q "your-openai-api-key-here" .env; then
    echo ""
    echo "============================================"
    echo "⚠️  SETUP REQUIRED"
    echo "============================================"
    echo ""
    echo "Before running the chatbot, you need to:"
    echo ""
    echo "1. Edit .env and add your API keys:"
    echo "   - OPENAI_API_KEY"
    echo "   - STRIPE_SECRET_KEY (optional)"
    echo "   - TWILIO_ACCOUNT_SID (optional)"
    echo "   - SMTP credentials (optional)"
    echo ""
    echo "2. Run: nano .env (or use your editor)"
    echo ""
    echo "3. After configuring, start the server:"
    echo "   python3 main.py"
    echo ""
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "✅ READY TO START"
    echo "============================================"
    echo ""
    echo "Run the following command to start:"
    echo ""
    echo "  python3 main.py"
    echo ""
    echo "Or for development with auto-reload:"
    echo ""
    echo "  uvicorn main:app --reload"
    echo ""
    echo "API will be available at:"
    echo "  http://localhost:8000"
    echo ""
    echo "API Documentation:"
    echo "  http://localhost:8000/docs"
    echo ""
    echo "============================================"
fi

echo ""
echo "🎉 Setup complete!"