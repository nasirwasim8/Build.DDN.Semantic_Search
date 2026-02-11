# GCP Deployment - Quick Start

## 🚀 Deploy in 3 Steps

### 1. Install Tools
```bash
# Install Google Cloud CLI
brew install --cask google-cloud-sdk

# Install Firebase CLI
npm install -g firebase-tools

# Login
gcloud auth login
firebase login
```

### 2. Create GCP Project
```bash
# Create project
gcloud projects create YOUR-PROJECT-ID

# Set as active
gcloud config set project YOUR-PROJECT-ID

# Enable billing (required - use free $300 credit)
# Go to: https://console.cloud.google.com/billing
```

### 3. Deploy!
```bash
cd Build.Semantic_Search
./deploy.sh
```

**Done!** Your app will be live at: `https://YOUR-PROJECT-ID.web.app`

---

## Environment Setup

Before deploying, set your INFINIA credentials:

```bash
# Edit this file
nano backend/.env.production

# Add:
INFINIA_ENDPOINT=https://your-endpoint.com
INFINIA_ACCESS_KEY=your-key
INFINIA_SECRET_KEY=your-secret
INFINIA_BUCKET=multimodal-search
```

---

## What Gets Deployed

**Backend → Cloud Run:**
- FastAPI application in Docker container
- Auto-scaling (0 to 10 instances)
- 2 GB RAM, 2 CPU cores
- HTTPS endpoint automatically

**Frontend → Firebase:**
- React app on global CDN
- Automatic SSL certificate
- Custom domain support

---

## Costs

**Free Tier:**
- 2 million Cloud Run requests/month
- 10 GB Firebase hosting

**Typical usage:** $0-10/month

---

## Need Help?

See full guide: `DEPLOYMENT_GUIDE.md`

**Common Issues:**
- Billing not enabled → Enable in console
- gcloud not found → Install Cloud CLI
- Build fails → Check `backend/Dockerfile`

---

**Your shareable link will be live in ~5 minutes!** 🎉
