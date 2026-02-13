#!/bin/bash

################################################################################
# Simplified Deploy - Build.DDN.Intelligence to Ubuntu Server
# Deploys to home directory (no sudo required)
################################################################################

set -e

# Configuration
APP_NAME="Build.DDN.Intelligence"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEPLOY_PACKAGE="${APP_NAME}_deploy_${TIMESTAMP}.tar.gz"

# Server Configuration
SERVER_USER="nwasim"
SERVER_HOST="10.36.97.158"
SERVER_PATH="/home/${SERVER_USER}/${APP_NAME}"  # Using home directory

# Port Configuration
BACKEND_PORT=8001
FRONTEND_PORT=5175

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# ============================================================================
# STEP 1: CREATE PACKAGE
# ============================================================================

print_header "Step 1: Creating Deployment Package"

tar -czf "${DEPLOY_PACKAGE}" \
  --exclude='*.tar.gz' \
  --exclude='*.zip' \
  --exclude='*_backup_*' \
  --exclude='*_deploy_*' \
  --exclude='cache_disabled' \
  --exclude='node_modules' \
  --exclude='venv' \
  --exclude='Sem-Search' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='dist' \
  --exclude='build' \
  --exclude='.vscode' \
  --exclude='.idea' \
  --exclude='*.log' \
  --exclude='.DS_Store' \
  --exclude='coverage' \
  --exclude='.next' \
  --exclude='*.egg-info' \
  --exclude='.vite' \
  --exclude='cache' \
  --exclude='*.mp4' \
  --exclude='*.avi' \
  --exclude='*.mov' \
  .

PACKAGE_SIZE=$(ls -lh "${DEPLOY_PACKAGE}" | awk '{print $5}')
print_success "Package created: ${DEPLOY_PACKAGE} (${PACKAGE_SIZE})"

# ============================================================================
# STEP 2: TRANSFER TO SERVER
# ============================================================================

print_header "Step 2: Transferring to Server"

scp "${DEPLOY_PACKAGE}" "${SERVER_USER}@${SERVER_HOST}:/tmp/"
print_success "Package transferred"

# ============================================================================
# STEP 3: DEPLOY ON SERVER
# ============================================================================

print_header "Step 3: Deploying on Server"

ssh "${SERVER_USER}@${SERVER_HOST}" << 'ENDSSH'
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

APP_NAME="Build.DDN.Intelligence"
SERVER_PATH="/home/nwasim/${APP_NAME}"
BACKEND_PORT=8001
FRONTEND_PORT=5175

echo -e "${BLUE}▶ Setting up application directory${NC}"

# Create directory (no sudo needed in home)
mkdir -p "${SERVER_PATH}"
cd "${SERVER_PATH}"

# Extract package
DEPLOY_PACKAGE=$(ls -t /tmp/${APP_NAME}_deploy_*.tar.gz | head -1)
echo -e "${BLUE}▶ Extracting ${DEPLOY_PACKAGE}${NC}"
tar -xzf "${DEPLOY_PACKAGE}"
rm "${DEPLOY_PACKAGE}"

echo -e "${GREEN}✓ Package extracted${NC}"

# ============================================================================
# BACKEND SETUP
# ============================================================================

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Backend Setup${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

cd "${SERVER_PATH}/backend"

# Create virtual environment
echo -e "${BLUE}▶ Creating Python virtual environment${NC}"
python3 -m venv venv

# Install dependencies
echo -e "${BLUE}▶ Installing Python dependencies${NC}"
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
deactivate

echo -e "${GREEN}✓ Backend dependencies installed${NC}"

# ============================================================================
# FRONTEND SETUP
# ============================================================================

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Frontend Setup${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

cd "${SERVER_PATH}/frontend"

# Install dependencies
echo -e "${BLUE}▶ Installing Node.js dependencies${NC}"
npm install --silent

# Build frontend
echo -e "${BLUE}▶ Building frontend${NC}"
npm run build

echo -e "${GREEN}✓ Frontend built${NC}"

# ============================================================================
# PM2 SETUP
# ============================================================================

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PM2 Configuration${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

# Check PM2
if ! command -v pm2 &> /dev/null; then
    echo -e "${YELLOW}⚠ Installing PM2 globally${NC}"
    npm install -g pm2
fi

# Stop existing instances
echo -e "${BLUE}▶ Stopping existing instances${NC}"
pm2 delete "${APP_NAME}-backend" 2>/dev/null || true
pm2 delete "${APP_NAME}-frontend" 2>/dev/null || true

# Create PM2 config
cat > "${SERVER_PATH}/ecosystem.config.js" << 'EOFPM2'
module.exports = {
  apps: [
    {
      name: "Build.DDN.Intelligence-backend",
      cwd: "/home/nwasim/Build.DDN.Intelligence/backend",
      interpreter: "/home/nwasim/Build.DDN.Intelligence/backend/venv/bin/python",
      script: "main.py",
      env: {
        PORT: 8001,
        HOST: "0.0.0.0"
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      error_file: "/home/nwasim/Build.DDN.Intelligence/logs/backend-error.log",
      out_file: "/home/nwasim/Build.DDN.Intelligence/logs/backend-out.log",
      log_file: "/home/nwasim/Build.DDN.Intelligence/logs/backend-combined.log",
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
      log_file: "/home/nwasim/Build.DDN.Intelligence/logs/frontend-combined.log",
      time: true
    }
  ]
};
EOFPM2

# Create logs directory
mkdir -p "${SERVER_PATH}/logs"

echo -e "${GREEN}✓ PM2 configuration created${NC}"

# ============================================================================
# START APPLICATIONS
# ============================================================================

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Starting Applications${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

cd "${SERVER_PATH}"
pm2 start ecosystem.config.js
pm2 save

echo -e "${GREEN}✓ Applications started${NC}"

# ============================================================================
# SUMMARY
# ============================================================================

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Deployment Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""
pm2 list
echo ""
echo -e "${GREEN}✓ Deployment successful!${NC}"
echo ""
echo -e "${BLUE}Access URLs:${NC}"
echo -e "  Backend:  http://10.36.97.158:8001"
echo -e "  Frontend: http://10.36.97.158:5175"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo -e "  View logs:    pm2 logs Build.DDN.Intelligence"
echo -e "  Restart:      pm2 restart Build.DDN.Intelligence"
echo -e "  Stop:         pm2 stop Build.DDN.Intelligence"
echo ""

ENDSSH

# ============================================================================
# CLEANUP
# ============================================================================

print_header "Step 4: Cleanup"
rm "${DEPLOY_PACKAGE}"
print_success "Local cleanup complete"

# ============================================================================
# FINAL MESSAGE
# ============================================================================

print_header "🎉 Deployment Complete!"

echo ""
print_success "Application deployed to ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}"
echo ""
echo -e "${GREEN}Access your application:${NC}"
echo -e "  Backend:  ${BLUE}http://${SERVER_HOST}:${BACKEND_PORT}${NC}"
echo -e "  Frontend: ${BLUE}http://${SERVER_HOST}:${FRONTEND_PORT}${NC}"
echo -e "  API Docs: ${BLUE}http://${SERVER_HOST}:${BACKEND_PORT}/docs${NC}"
echo ""
echo -e "${YELLOW}To view logs, SSH to server and run:${NC}"
echo -e "  ${BLUE}ssh ${SERVER_USER}@${SERVER_HOST}${NC}"
echo -e "  ${BLUE}pm2 logs Build.DDN.Intelligence${NC}"
echo ""
