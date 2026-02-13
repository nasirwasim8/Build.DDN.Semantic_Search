#!/bin/bash

# Push latest changes to GitHub
# Repository: https://github.com/nasirwasim8/Build.DDN.Semantic_Search

set -e

cd /Users/nwasim/Documents/MyDocs/llm_engineering-main/DDN/Infinia/kafka-pipeline/python-pipeline/Build.DDN.Com/Build.Semantic_Search

echo "========================================
📤 Pushing to GitHub
========================================"

# Check git status
echo "📋 Current status:"
git status

echo ""
echo "📝 Adding changes..."
git add .

echo ""
echo "💬 Creating commit..."
git commit -m "Fix: Image loading and video upload issues

- Added /api/browse/image-stream/ endpoint to proxy image requests
- Fixed browser certificate issues with INFINIA self-signed SSL
- Removed broken helper function references in video processing
- Updated .gitignore to exclude backups and cache directories
- Added deployment scripts for Ubuntu server

Deployed to production on port 5175"

echo ""
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "========================================
✅ Successfully pushed to GitHub!
========================================"
echo ""
echo "Repository: https://github.com/nasirwasim8/Build.DDN.Semantic_Search"
