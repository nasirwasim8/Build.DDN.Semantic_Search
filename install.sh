#!/bin/bash

# DDN Multimodal Semantic Search - Installation Script
# =====================================================

set -e

echo "=================================================="
echo "  DDN Multimodal Semantic Search - Setup"
echo "  Powered by DDN INFINIA & NVIDIA CLIP/BLIP"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    else
        echo -e "${RED}✗ Python 3.9+ required, found $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi

# Check Node.js version
echo "Checking Node.js version..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -ge 18 ]; then
        echo -e "${GREEN}✓ Node.js v$(node -v | cut -d'v' -f2) found${NC}"
    else
        echo -e "${RED}✗ Node.js 18+ required, found v$(node -v)${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Node.js not found${NC}"
    exit 1
fi

# Check npm
echo "Checking npm..."
if command -v npm &> /dev/null; then
    echo -e "${GREEN}✓ npm $(npm -v) found${NC}"
else
    echo -e "${RED}✗ npm not found${NC}"
    exit 1
fi

echo ""
echo "=================================================="
echo "  Installing Backend Dependencies"
echo "=================================================="
echo ""

cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}! Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "Installing Python dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet

echo -e "${GREEN}✓ Backend dependencies installed${NC}"

# Deactivate virtual environment
deactivate

cd ..

echo ""
echo "=================================================="
echo "  Installing Frontend Dependencies"
echo "=================================================="
echo ""

cd frontend

# Install npm dependencies
echo "Installing npm dependencies..."
npm install --silent

echo -e "${GREEN}✓ Frontend dependencies installed${NC}"

cd ..

echo ""
echo "=================================================="
echo -e "${GREEN}  Installation Complete!${NC}"
echo "=================================================="
echo ""
echo "To start the application:"
echo ""
echo "  1. Start the backend (Terminal 1):"
echo "     cd backend"
echo "     source venv/bin/activate"
echo "     python main.py"
echo ""
echo "  2. Start the frontend (Terminal 2):"
echo "     cd frontend"
echo "     npm run dev"
echo ""
echo "  3. Open your browser:"
echo "     Frontend: http://localhost:5173"
echo "     API Docs: http://localhost:8000/docs"
echo ""
echo "=================================================="
echo ""
