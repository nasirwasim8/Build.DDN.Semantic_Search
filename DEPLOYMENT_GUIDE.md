# Deployment Guide - Build.DDN.Intelligence

## Quick Deployment

### One-Command Deployment from macOS

```bash
./deploy_to_ubuntu.sh
```

That's it! The script will:
1. ✅ Create a clean deployment package (excludes backups, cache_disabled, node_modules, etc.)
2. ✅ Transfer to Ubuntu server at 10.36.97.158
3. ✅ Set up backend and frontend
4. ✅ Install all dependencies
5. ✅ Configure PM2 for daemon-style operation
6. ✅ Start services on the correct ports

---

## Configuration

### Application Details
- **Name:** Build.DDN.Intelligence
- **Server:** nwasim@10.36.97.158
- **Location:** /opt/Build.DDN.Intelligence
- **Backend Port:** 8001
- **Frontend Port:** 5175

### What's Deployed
✅ Source code (backend, frontend)  
✅ Configuration files  
✅ Documentation  
✅ Deployment scripts  

### What's Excluded
❌ Backup/archive files (*.tar.gz, *.zip)  
❌ Previous deployment packages  
❌ cache_disabled folder  
❌ node_modules  
❌ venv and Sem-Search (virtual environments)  
❌ Video files (*.mp4, *.avi, *.mov)  
❌ Git repository (.git)  
❌ __pycache__, .git, etc.  

---

## PM2 Management

Once deployed, the application runs as a PM2-managed service. This means:
- ✅ Auto-restarts on crash
- ✅ Survives SSH disconnection
- ✅ Starts on server reboot
- ✅ Centralized logging

### Useful PM2 Commands

SSH to the server first:
```bash
ssh nwasim@10.36.97.158
```

Then use these commands:

```bash
# View all running applications
pm2 list

# View real-time logs
pm2 logs Build.DDN.Intelligence

# View backend logs only
pm2 logs Build.DDN.Intelligence-backend

# View frontend logs only
pm2 logs Build.DDN.Intelligence-frontend

# Restart the application
pm2 restart Build.DDN.Intelligence

# Stop the application
pm2 stop Build.DDN.Intelligence

# Start the application
pm2 start Build.DDN.Intelligence

# Delete the application from PM2
pm2 delete Build.DDN.Intelligence

# Monitor resources
pm2 monit
```

---

## Access URLs

After deployment, access the application at:

- **Backend API:** http://10.36.97.158:8001
- **Frontend UI:** http://10.36.97.158:5175
- **API Docs:** http://10.36.97.158:8001/docs

---

## Troubleshooting

### Check if services are running
```bash
ssh nwasim@10.36.97.158
pm2 list
```

### View error logs
```bash
pm2 logs Build.DDN.Intelligence --err
```

### Restart services
```bash
pm2 restart Build.DDN.Intelligence
```

### Check port availability
```bash
# Check if ports are already in use
sudo netstat -tulpn | grep -E '8001|5175'
```

### Manual start (if needed)
```bash
cd /opt/Build.DDN.Intelligence

# Backend
cd backend
source venv/bin/activate
python main.py

# Frontend (in a new terminal)
cd frontend
npm run preview -- --host 0.0.0.0 --port 5175
```

---

## Re-deployment

To redeploy with updates, simply run the script again:

```bash
./deploy_to_ubuntu.sh
```

The script will:
1. Create a fresh package
2. Stop existing PM2 services
3. Extract new code
4. Reinstall dependencies
5. Restart services

---

## Log Files

PM2 logs are stored at:
```
/opt/Build.DDN.Intelligence/logs/
├── backend-error.log
├── backend-out.log
├── backend-combined.log
├── frontend-error.log
├── frontend-out.log
└── frontend-combined.log
```

---

## Notes

- The application runs as a daemon service (survives SSH disconnection)
- PM2 automatically restarts the app if it crashes
- Logs are rotated automatically by PM2
- Services start automatically on server reboot
