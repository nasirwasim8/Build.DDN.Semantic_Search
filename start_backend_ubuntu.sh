#!/bin/bash

# Quick fix to start the backend on Ubuntu server
# Run this on the server after deployment fails

set -e

echo "🔧 Starting Backend Service..."

cd /home/nwasim/Build.DDN.Intelligence

# Update ecosystem.config.js with correct configuration
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

# Delete any failed backend processes
pm2 delete Build.DDN.Intelligence-backend 2>/dev/null || true

# Start backend
echo "Starting backend..."
pm2 start ecosystem.config.js --only Build.DDN.Intelligence-backend

# Save configuration
pm2 save

# Show status
echo ""
pm2 list

echo ""
echo "📝 Backend logs:"
pm2 logs Build.DDN.Intelligence-backend --lines 20 --nostream

echo ""
echo "✅ Backend started!"
echo "🌐 Access your app at: http://10.36.97.158:5175"
