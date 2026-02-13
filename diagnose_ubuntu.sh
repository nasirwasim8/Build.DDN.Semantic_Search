#!/bin/bash

# Run this script ON THE UBUNTU SERVER to diagnose backend crash issue
# ssh nwasim@10.36.97.158
# cd ~/Build.DDN.Intelligence
# chmod +x diagnose.sh
# ./diagnose.sh

echo "========================================"
echo "🔍 Diagnosing Backend Issues"
echo "========================================"
echo ""

echo "📊 PM2 Status:"
pm2 list

echo ""
echo "========================================"
echo "📝 Backend Logs (last 100 lines):"
echo "========================================"
pm2 logs 4 --lines 100 --nostream

echo ""
echo "========================================"
echo "🔍 Backend Details:"
echo "========================================"
pm2 describe 4

echo ""
echo "========================================"
echo "🔍 Checking Python Environment:"
echo "========================================"
cd /home/nwasim/Build.DDN.Intelligence/backend
which python3
python3 --version
echo ""
echo "Checking if main.py exists:"
ls -la main.py

echo ""
echo "========================================"
echo "🔍 Checking Storage Config:"
echo "========================================"
cat data/storage_config.json | grep endpoint_url

echo ""
echo "Done!"
