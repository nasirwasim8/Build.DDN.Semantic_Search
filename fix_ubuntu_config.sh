#!/bin/bash

# Fix Ubuntu server storage config to use IP address instead of hostname
# This prevents image loading issues

SERVER="nwasim@10.36.97.158"

echo "🔧 Fixing storage config on Ubuntu server..."
echo ""

ssh "${SERVER}" << 'ENDSSH'
cd /home/nwasim/Build.DDN.Intelligence/backend/data

# Backup current config
cp storage_config.json storage_config.json.backup

# Replace hostname with IP address
sed -i 's|https://ddn-ai-demo-env:8111|https://10.36.111.120:8111|g' storage_config.json

echo "✅ Updated storage_config.json:"
cat storage_config.json | grep endpoint_url

echo ""
echo "🔄 Restarting backend to apply changes..."
cd /home/nwasim/Build.DDN.Intelligence
pm2 restart backend

echo ""
echo "📝 Backend logs:"
pm2 logs backend --lines 10 --nostream

ENDSSH

echo ""
echo "✅ Configuration fixed!"
echo "Images should now load properly on Ubuntu server"
