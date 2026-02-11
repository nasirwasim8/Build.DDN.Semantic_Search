#!/bin/bash
# Setup script for Celery video processing

set -e

echo "🚀 Setting up async video processing infrastructure..."
echo ""

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo "❌ Redis not found. Installing with Homebrew..."
    
    if command -v brew &> /dev/null; then
        brew install redis
        echo "✅ Redis installed"
    else
        echo "❌ Homebrew not found. Please install Redis manually:"
        echo "   Visit: https://redis.io/docs/getting-started/installation/install-redis-on-mac-os/"
        exit 1
    fi
else
    echo "✅ Redis already installed"
fi

# Start Redis server in background if not running
if ! redis-cli ping &> /dev/null; then
    echo "🔄 Starting Redis server..."
    brew services start redis || redis-server --daemonize yes
    sleep 2
    echo "✅ Redis started"
else
    echo "✅ Redis already running"
fi

# Verify Redis connection
if redis-cli ping &> /dev/null; then
    echo "✅ Redis connection verified (PONG)"
else
    echo "❌ Redis connection failed"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Start Celery worker:"
echo "      cd backend"
echo "      celery -A app.celery_app worker --loglevel=info -Q video_processing,image_processing"
echo ""
echo "   2. In another terminal, start FastAPI (if not running):"
echo "      cd backend"
echo "      uvicorn app.main:app --reload"
echo ""
echo "   3. Try uploading a video again!"
