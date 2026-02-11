# 🚀 Demo Mode - Local Cache Configuration

## Quick Toggle for Conference Demo

### **Enable Demo Mode (Local Cache Only)**
For fast, offline demo performance at the conference.

**File:** `backend/app/services/storage.py`  
**Line 8:**
```python
USE_LOCAL_CACHE = True  # ✅ DEMO MODE - No S3 calls, instant performance
```

### **Disable Demo Mode (Back to Production)**
After the conference, revert to normal S3 operation.

**File:** `backend/app/services/storage.py`  
**Line 8:**
```python
USE_LOCAL_CACHE = False  # ❌ PRODUCTION MODE - Uses S3 buckets
```

---

## What Demo Mode Does

When `USE_LOCAL_CACHE = True`:
- ✅ All videos load from `cache/videos/` folder (instant)
- ✅ All metadata loads from `cache/embeddings/` JSON files (instant)
- ✅ Search works from local cache only (no network calls)
- ✅ Video streaming is instant (local disk)
- ❌ **NO S3 calls at all** - completely offline capable

When `USE_LOCAL_CACHE = False`:
- S3 operations resume normally
- Videos stream from Infinia S3 bucket
- Metadata fetched from S3 headers

---

## Current Cache Contents

📁 **Videos:** 6 MP4 files in `cache/videos/`
- `20251214_201255_Car burglar_1.mp4`
- `20251214_201358_Car burglar_2.mp4`
- `20251214_201449_Car burglar_3.mp4`
- `20251215_122637_Shoplifting_2.mp4`
- `20251215_191541_Jensen_Alex.mp4` ⭐ (Alex)
- `20260126_104339_Retail_Store_CCTV_1.mp4`

📄 **Metadata:** 51 JSON files in `cache/embeddings/`

---

## How to Restart Backend

After changing the flag:

```bash
# If using systemd/background process, restart it
# OR just restart the Python backend server

cd /path/to/Build.Semantic_Search
# Kill existing backend process, then:
python -m uvicorn backend.app.main:app --reload
```

---

## Performance Comparison

| Operation | S3 Mode | Demo Mode (Local Cache) |
|-----------|---------|-------------------------|
| List videos | ~500-2000ms | **~5ms** ⚡ |
| Load video | ~1000-5000ms | **~20ms** ⚡ |
| Search metadata | ~800-3000ms | **~10ms** ⚡ |

**Result:** 50-100x faster for conference demo! 🎯

---

## Reverting After Conference

1. Open `backend/app/services/storage.py`
2. Change line 8: `USE_LOCAL_CACHE = False`
3. Restart backend server
4. Done! Back to normal S3 operation

**That's it!** Single line change, instant toggle.
