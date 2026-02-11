#!/bin/bash

# Clean Backup Script - Excludes node_modules, venv, cache, and other build artifacts
# Usage: ./create_backup.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="Build.Semantic_Search_backup_${TIMESTAMP}.tar.gz"
SOURCE_DIR="."
DEST_DIR=".."

echo "🔄 Creating backup: ${BACKUP_NAME}"
echo "📦 Source: $(pwd)"
echo ""

# Create backup excluding large directories and build artifacts
tar -czf "${DEST_DIR}/${BACKUP_NAME}" \
  --exclude='node_modules' \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='cache' \
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
  "${SOURCE_DIR}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Backup created successfully!"
    echo "📍 Location: ${DEST_DIR}/${BACKUP_NAME}"
    echo ""
    ls -lh "${DEST_DIR}/${BACKUP_NAME}"
    echo ""
    echo "📋 What's included:"
    echo "   ✓ Source code (backend, frontend)"
    echo "   ✓ Configuration files"
    echo "   ✓ Documentation"
    echo "   ✓ Deployment scripts"
    echo ""
    echo "❌ What's excluded:"
    echo "   ✗ node_modules/"
    echo "   ✗ venv/"
    echo "   ✗ cache/"
    echo "   ✗ __pycache__/"
    echo "   ✗ .git/"
else
    echo ""
    echo "❌ Backup failed!"
    exit 1
fi
