# Video Processing Setup Guide

## Issue Diagnosed

✅ **Video upload works** - file uploaded to Infinia successfully  
❌ **Processing stuck** - Celery worker not running, tasks never execute  
❌ **Redis not installed** - required for Celery task queue

---

## Quick Start (3 Steps)

### Step 1: Install and Start Redis

```bash
# Run the automated setup script
./setup_celery.sh
```

**OR manually:**

```bash
# Install Redis
brew install redis

# Start Redis
brew services start redis

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### Step 2: Install Python Dependencies (if not done)

```bash
cd backend
pip install celery redis kombu
```

### Step 3: Start Celery Worker

**Open a NEW terminal window** and run:

```bash
cd backend
celery -A app.celery_app worker --loglevel=info -Q video_processing,image_processing
```

You should see:
```
-------------- celery@yourmac v5.3.x
--- ***** -----
-- ******* ---- Darwin
- *** --- * ---
- ** ---------- [config]
- ** ---------- .> app:         app.celery_app
- ** ---------- .> transport:   redis://localhost:6379/0
- ** ---------- .> results:     redis://localhost:6379/0
- *** --- * --- .> concurrency: 8 (prefork)
-- ******* ----
--- ***** -----
-------------- [queues]
.> video_processing exchange=video_processing(direct) key=video_processing
.> image_processing exchange=image_processing(direct) key=image_processing

[tasks]
  . app.tasks.video_tasks.process_video_task
```

### Step 4: Try Upload Again

1. Keep Celery worker running in that terminal
2. Go back to the browser
3. Upload a video → should now process!

---

## Terminal Setup

You need **3 terminals running simultaneously**:

### Terminal 1: FastAPI Backend (already running)
```bash
cd backend
uvicorn app.main:app --reload
```

### Terminal 2: Redis Server (new - auto-start with brew services)
```bash
# If you used brew services start redis, this runs in background
# Check status:
brew services list | grep redis
```

### Terminal 3: Celery Worker (new - REQUIRED)
```bash
cd backend
celery -A app.celery_app worker --loglevel=info -Q video_processing,image_processing
```

---

## Verification Checklist

Before uploading video, verify:

- [ ] **Redis running**: `redis-cli ping` returns `PONG`
- [ ] **FastAPI running**: `curl http://localhost:8000/health` returns JSON
- [ ] **Celery worker running**: Terminal shows "celery@yourmac ready"

---

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'celery'"
**Fix:** Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Issue: "Cannot connect to redis://localhost:6379"
**Fix:** Start Redis
```bash
brew services start redis
```

### Issue: Task stuck in "pending" forever
**Fix:** Celery worker not running. Start worker in Terminal 3 (see above)

### Issue: "Cannot find module app.celery_app"
**Fix:** Make sure you're in the `backend` directory when running celery command

---

## Stopping Services

```bash
# Stop Redis
brew services stop redis

# Stop Celery worker
# In the Celery terminal, press Ctrl+C

# Stop FastAPI
# In the FastAPI terminal, press Ctrl+C
```

---

## Alternative: Using Docker Compose (Future Enhancement)

For production deployment, consider creating a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  celery-worker:
    build: ./backend
    command: celery -A app.celery_app worker --loglevel=info
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
```

Then simply: `docker-compose up`

---

## Environment Variables

Add to your `.env` file:

```bash
# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Video Processing
VIDEO_CHUNK_DURATION=10.0
KEYFRAME_FPS=1.0
BATCH_SIZE=8

# GPU Configuration (0 = CPU only)
GPU_COUNT=0
```

---

## Testing After Setup

1. **Upload a small test video** (keep it under 30 seconds for testing)
2. **Watch Celery worker terminal** - you should see:
   ```
   [2024-02-10 21:45:00] Received task: app.tasks.video_tasks.process_video_task
   [2024-02-10 21:45:01] 🎬 Processing video <asset_id> on GPU 0
   [2024-02-10 21:45:02] 📥 Downloaded video to /tmp/...
   [2024-02-10 21:45:03] 📊 Video: 1920x1080, 15.2s, 2 chunks
   [2024-02-10 21:45:10] Task completed successfully
   ```
3. **Watch browser** - status should change: pending → processing → completed
4. **Expand metadata** - should show AI summary, detected objects, custom tags

---

## Quick Troubleshooting

Run this command to check all services:

```bash
echo "=== Service Status ==="
echo -n "Redis: " && redis-cli ping 2>&1
echo -n "FastAPI: " && curl -s http://localhost:8000/health | jq -r '.status' 2>&1 || echo "Not running"
echo -n "Celery: " && celery -A app.celery_app inspect ping 2>&1 | grep -q "pong" && echo "Running" || echo "Not running"
```

Good luck! 🚀
