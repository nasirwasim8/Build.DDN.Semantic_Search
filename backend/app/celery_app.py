"""
Celery application configuration for async video processing.

This module configures Celery for background task processing with:
- Redis broker for message queue
- Result backend for status tracking  
- GPU-aware task routing
- Retry policies and error handling
"""

import os
import logging
from celery import Celery
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    'semantic_search',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Celery Configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        'app.tasks.video_tasks.*': {'queue': 'video_processing'},
        'app.tasks.image_tasks.*': {'queue': 'image_processing'},
    },
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Task execution
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for GPU workloads
    worker_max_tasks_per_child=10,  # Restart worker after N tasks (prevent memory leaks)
    
    # Retry settings
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,
    
    # Performance
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
)

# Task discovery
celery_app.autodiscover_tasks(['app.tasks'])

logger.info(f"✅ Celery configured with broker: {settings.CELERY_BROKER_URL}")
logger.info(f"✅ Result backend: {settings.CELERY_RESULT_BACKEND}")
