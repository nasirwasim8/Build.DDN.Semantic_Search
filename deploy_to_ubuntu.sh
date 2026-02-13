#!/bin/bash

# Deployment Script for Build.DDN.Intelligence
# This script syncs code changes to Ubuntu server and restarts PM2 services
# You'll be prompted for your SSH password 3 times (backend sync, frontend sync, PM2 restart)

set -e  # Exit on error

# Configuration
SERVER="nwasim@10.36.97.158"
LOCAL_DIR="/Users/nwasim/Documents/MyDocs/llm_engineering-main/DDN/Infinia/kafka-pipeline/python-pipeline/Build.DDN.Com/Build.Semantic_Search"
REMOTE_DIR="/home/nwasim/Build.DDN.Intelligence"

echo "========================================"
echo "🚀 Deploying to Ubuntu Server"
echo "========================================"
echo ""

# Step 1: Sync Backend
echo "📦 Syncing backend files..."
rsync -avz --progress \
  --exclude='node_modules' \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='*.pyc' \
  --exclude='data/storage_config.json' \
  "${LOCAL_DIR}/backend/" \
  "${SERVER}:${REMOTE_DIR}/backend/"

echo ""
echo "✅ Backend synced successfully"
echo ""

# Step 2: Sync Frontend
echo "📦 Syncing frontend files..."
rsync -avz --progress \
  --exclude='node_modules' \
  --exclude='.next' \
  --exclude='dist' \
  --exclude='build' \
  "${LOCAL_DIR}/frontend/src/" \
  "${SERVER}:${REMOTE_DIR}/frontend/src/"

echo ""
echo "✅ Frontend synced successfully"
echo ""

# Step 3: Restart PM2 Services
echo "🔄 Restarting PM2 services on server..."
ssh "${SERVER}" << 'ENDSSH'
cd /home/nwasim/Build.DDN.Intelligence

echo "Restarting backend..."
pm2 restart backend

echo "Restarting frontend..."
pm2 restart frontend

echo ""
echo "📊 PM2 Status:"
pm2 list

echo ""
echo "📝 Recent logs:"
pm2 logs --lines 10 --nostream

ENDSSH

echo ""
echo "========================================"
echo "✅ Deployment Complete!"
echo "========================================"
echo ""
echo "🌐 Access your app at: http://10.36.97.158:5175"
echo ""
echo "To view logs on server:"
echo "  ssh ${SERVER}"
echo "  pm2 logs backend --lines 50"
echo "  pm2 logs frontend --lines 50"
echo ""
