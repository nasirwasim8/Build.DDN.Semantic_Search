#!/bin/bash

# Disk Space Cleanup Script for Ubuntu Server
# Run this on the Ubuntu server to free up space

echo "========================================
🧹 Cleaning Up Disk Space
========================================"

# Check current disk usage
echo "Current disk usage:"
df -h /

echo ""
echo "Cleaning up..."

# 1. Clean apt cache
echo "1. Cleaning apt cache..."
sudo apt-get clean
sudo apt-get autoclean

# 2. Remove old kernels (keep current and one previous)
echo "2. Removing old kernels..."
sudo apt-get autoremove --purge -y

# 3. Clean pip cache
echo "3. Cleaning pip cache..."
pip cache purge 2>/dev/null || true
python3 -m pip cache purge 2>/dev/null || true

# 4. Clean ML model caches (Hugging Face, PyTorch, CLIP)
echo "4. Cleaning ML model caches..."
echo "   → Removing Hugging Face cache..."
rm -rf ~/.cache/huggingface/ 2>/dev/null || true
echo "   → Removing PyTorch cache..."
rm -rf ~/.cache/torch/ 2>/dev/null || true
echo "   → Removing CLIP cache..."
rm -rf ~/.cache/clip/ 2>/dev/null || true
echo "   ✓ ML caches cleaned"

# 5. Clean npm cache
echo "5. Cleaning npm cache..."
npm cache clean --force 2>/dev/null || true

# 6. Remove old PM2 logs (keep last 7 days)
echo "6. Removing old PM2 logs..."
find ~/.pm2/logs -type f -mtime +7 -delete 2>/dev/null || true

# 7. Clean journalctl logs (keep last 3 days)
echo "7. Cleaning system logs..."
sudo journalctl --vacuum-time=3d

# 8. Remove old backups if any exist
echo "8. Removing old backups..."
cd /home/nwasim
rm -rf Build.DDN.Intelligence.backup_* 2>/dev/null || true
rm -rf Build.DDN.RAG.backup_* 2>/dev/null || true

# 9. Clean tmp directory
echo "9. Cleaning /tmp..."
sudo find /tmp -type f -atime +7 -delete 2>/dev/null || true

# 10. Remove Python __pycache__ directories
echo "10. Removing Python cache..."
find /home/nwasim -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /home/nwasim -type f -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "========================================
✅ Cleanup Complete!
========================================"
echo ""
echo "Disk usage after cleanup:"
df -h /

echo ""
echo "Largest directories in /home/nwasim:"
du -sh /home/nwasim/* 2>/dev/null | sort -hr | head -10
