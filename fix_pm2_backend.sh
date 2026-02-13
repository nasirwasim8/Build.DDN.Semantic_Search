#!/bin/bash

# Fix PM2 backend configuration on Ubuntu server
# Run this ON THE UBUNTU SERVER

echo "🔧 Stopping backend with reload loop..."
pm2 stop 4

echo ""
echo "🔧 Deleting old PM2 process..."
pm2 delete 4

echo ""
echo "🚀 Starting backend in PRODUCTION mode (no reload)..."
cd /home/nwasim/Build.DDN.Intelligence/backend

# Start without --reload flag for production
pm2 start python3 \
  --name "Build.DDN.Intelligence-backend" \
  --interpreter none \
  -- -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8001

echo ""
echo "💾 Saving PM2 configuration..."
pm2 save

echo ""
echo "📊 PM2 Status:"
pm2 list

echo ""
echo "📝 Backend logs:"
pm2 logs Build.DDN.Intelligence-backend --lines 20 --nostream

echo ""
echo "✅ Done! Backend should now be stable."
