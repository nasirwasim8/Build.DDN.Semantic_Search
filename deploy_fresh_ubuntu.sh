#!/bin/bash

# Fresh Deployment Script for Build.DDN.Intelligence
# This script will backup existing deployment, sync fresh code, and configure PM2

set -e

SERVER="nwasim@10.36.97.158"
LOCAL_DIR="/Users/nwasim/Documents/MyDocs/llm_engineering-main/DDN/Infinia/kafka-pipeline/python-pipeline/Build.DDN.Com/Build.Semantic_Search"
REMOTE_DIR="/home/nwasim/Build.DDN.Intelligence"

echo "========================================
🚀 Fresh Deployment to Ubuntu Server
========================================"

# Step 1: Stop PM2 services and clean existing deployment
echo "🛑 Stopping services and cleaning existing deployment..."
ssh "${SERVER}" << 'ENDSSH'
cd /home/nwasim

# Stop PM2 processes
pm2 stop Build.DDN.Intelligence-backend 2>/dev/null || true
pm2 stop Build.DDN.Intelligence-frontend 2>/dev/null || true
pm2 delete Build.DDN.Intelligence-backend 2>/dev/null || true
pm2 delete Build.DDN.Intelligence-frontend 2>/dev/null || true

# Preserve storage config if it exists
if [ -f "Build.DDN.Intelligence/backend/data/storage_config.json" ]; then
    echo "Preserving storage_config.json..."
    mkdir -p /tmp/build_ddn_backup
    cp Build.DDN.Intelligence/backend/data/storage_config.json /tmp/build_ddn_backup/
fi

# Delete existing deployment
if [ -d "Build.DDN.Intelligence" ]; then
    echo "Deleting existing deployment to save disk space..."
    rm -rf Build.DDN.Intelligence
fi

# Create fresh directory structure
mkdir -p Build.DDN.Intelligence/{backend/data,frontend,logs}

# Restore storage config
if [ -f "/tmp/build_ddn_backup/storage_config.json" ]; then
    echo "Restoring storage_config.json..."
    cp /tmp/build_ddn_backup/storage_config.json Build.DDN.Intelligence/backend/data/
    rm -rf /tmp/build_ddn_backup
fi
ENDSSH

# Step 2: Sync backend code
echo ""
echo "📦 Syncing backend files..."
rsync -avz --progress \
  --exclude='node_modules' \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='*.pyc' \
  --exclude='data/storage_config.json' \
  --exclude='data/cache_disabled' \
  --exclude='logs' \
  --exclude='backups' \
  --exclude='*.backup' \
  "${LOCAL_DIR}/backend/" \
  "${SERVER}:${REMOTE_DIR}/backend/"

# Step 3: Sync frontend code  
echo ""
echo "📦 Syncing frontend files..."
rsync -avz --progress \
  --exclude='node_modules' \
  --exclude='.next' \
  --exclude='dist' \
  --exclude='build' \
  --exclude='.cache' \
  --exclude='backups' \
  "${LOCAL_DIR}/frontend/" \
  "${SERVER}:${REMOTE_DIR}/frontend/"

# Step 4: Install dependencies and configure PM2 on server
echo ""
echo "🔧 Installing dependencies and configuring PM2..."
ssh "${SERVER}" << 'ENDSSH'
cd /home/nwasim/Build.DDN.Intelligence

# Install backend dependencies in venv
echo "Installing Python dependencies..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd ../frontend
npm install

# Build frontend for production
echo "Building frontend..."
npm run build

# Create logs directory
mkdir -p ../logs

# Create ecosystem.config.js
echo "Creating PM2 configuration..."
cd ..
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: "Build.DDN.Intelligence-backend",
      cwd: "/home/nwasim/Build.DDN.Intelligence/backend",
      script: "main.py",
      interpreter: "/home/nwasim/Build.DDN.Intelligence/backend/venv/bin/python3",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      error_file: "/home/nwasim/Build.DDN.Intelligence/logs/backend-error.log",
      out_file: "/home/nwasim/Build.DDN.Intelligence/logs/backend-out.log",
      time: true
    },
    {
      name: "Build.DDN.Intelligence-frontend",
      cwd: "/home/nwasim/Build.DDN.Intelligence/frontend",
      script: "npx",
      args: "vite preview --host 0.0.0.0 --port 5175",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "512M",
      error_file: "/home/nwasim/Build.DDN.Intelligence/logs/frontend-error.log",
      out_file: "/home/nwasim/Build.DDN.Intelligence/logs/frontend-out.log",
      time: true
    }
  ]
};
EOF

# Stop any existing processes
echo "Stopping existing PM2 processes..."
pm2 delete Build.DDN.Intelligence-backend 2>/dev/null || true
pm2 delete Build.DDN.Intelligence-frontend 2>/dev/null || true

# Start new processes
echo "Starting services..."
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Display status
echo ""
echo "========================================
📊 Deployment Status
========================================"
pm2 list

echo ""
echo "📝 Backend logs (last 20 lines):"
pm2 logs Build.DDN.Intelligence-backend --lines 20 --nostream

echo ""
echo "✅ Deployment complete!"
echo "🌐 Access your app at: http://10.36.97.158:5175"

ENDSSH

echo ""
echo "========================================
✅ Fresh Deployment Complete!
========================================"
echo ""
echo "🌐 Access your app at: http://10.36.97.158:5175"
echo ""
echo "To view logs:"
echo "  ssh ${SERVER}"
echo "  pm2 logs Build.DDN.Intelligence-backend"
echo "  pm2 logs Build.DDN.Intelligence-frontend"
echo ""
