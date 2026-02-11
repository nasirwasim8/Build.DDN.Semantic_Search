#!/bin/bash

# Deploy Backend to Cloud Run and Frontend to Firebase
# Usage: ./deploy.sh

set -e  # Exit on error

echo "🚀 Starting deployment to Google Cloud Platform..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if firebase is installed
if ! command -v firebase &> /dev/null; then
    echo "❌ Firebase CLI is not installed. Installing now..."
    npm install -g firebase-tools
fi

# Get project ID
echo -e "${BLUE}📋 Getting GCP project ID...${NC}"
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No GCP project selected. Please run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi
echo -e "${GREEN}✅ Using project: $PROJECT_ID${NC}"
echo ""

# Enable required APIs
echo -e "${BLUE}🔧 Enabling required Google Cloud APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    --project=$PROJECT_ID
echo -e "${GREEN}✅ APIs enabled${NC}"
echo ""

# Deploy Backend to Cloud Run
echo -e "${BLUE}🐳 Building and deploying backend to Cloud Run...${NC}"
cd backend

# Build Docker image
echo "Building Docker image..."
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/semantic-search-backend \
    --project=$PROJECT_ID

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy semantic-search-backend \
    --image gcr.io/$PROJECT_ID/semantic-search-backend \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --project=$PROJECT_ID

# Get backend URL
BACKEND_URL=$(gcloud run services describe semantic-search-backend \
    --platform managed \
    --region us-central1 \
    --format 'value(status.url)' \
    --project=$PROJECT_ID)

echo -e "${GREEN}✅ Backend deployed to: $BACKEND_URL${NC}"
echo ""
cd ..

# Update frontend API endpoint
echo -e "${BLUE}⚙️  Updating frontend API endpoint...${NC}"
cd frontend

# Create production environment file
cat > .env.production << EOF
VITE_API_URL=$BACKEND_URL
EOF

echo -e "${GREEN}✅ Frontend configured to use backend at: $BACKEND_URL${NC}"
echo ""

# Build frontend
echo -e "${BLUE}📦 Building frontend for production...${NC}"
npm install
npm run build
echo -e "${GREEN}✅ Frontend build complete${NC}"
echo ""

# Deploy to Firebase
echo -e "${BLUE}🔥 Deploying frontend to Firebase Hosting...${NC}"
firebase deploy --only hosting --project=$PROJECT_ID

# Get Firebase URL
FIREBASE_URL="https://$PROJECT_ID.web.app"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}🌐 Your application is live at:${NC}"
echo -e "${GREEN}   $FIREBASE_URL${NC}"
echo ""
echo -e "${BLUE}🔗 Backend API:${NC}"
echo -e "${GREEN}   $BACKEND_URL${NC}"
echo ""
echo -e "${YELLOW}⚠️  Next Steps:${NC}"
echo "1. Set environment variables in Cloud Run console"
echo "2. Configure CORS if needed"
echo "3. Set up custom domain (optional)"
echo ""
echo -e "${GREEN}Happy deploying! 🎉${NC}"

cd ..
