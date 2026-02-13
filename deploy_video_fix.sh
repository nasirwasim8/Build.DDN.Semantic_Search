#!/bin/bash

# Quick deployment script for video upload fix
# This fixes the "name '_process_video_with_chunks' is not defined" error

set -e

echo "========================================
🔧 Deploying Video Upload Fix
========================================"

# Copy fixed routes.py to server
echo "📦 Copying fixed routes.py..."
scp backend/app/api/routes.py nwasim@10.36.97.158:/home/nwasim/Build.DDN.Intelligence/backend/app/api/

echo ""
echo "🔄 Restarting backend..."
ssh nwasim@10.36.97.158 << 'ENDSSH'
cd /home/nwasim/Build.DDN.Intelligence
pm2 restart Build.DDN.Intelligence-backend
pm2 logs Build.DDN.Intelligence-backend --lines 20 --nostream
ENDSSH

echo ""
echo "========================================
✅ Video Upload Fix Deployed!
========================================"
echo ""
echo "Try uploading a video again at: http://10.36.97.158:5175"
