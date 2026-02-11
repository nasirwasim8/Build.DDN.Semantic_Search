#!/bin/bash
# Quick start script for Celery worker

cd backend

echo "🚀 Starting Celery worker for video processing..."
echo "   Press Ctrl+C to stop"
echo ""

celery -A app.celery_app worker \
  --loglevel=info \
  -Q video_processing,image_processing \
  --concurrency=2
