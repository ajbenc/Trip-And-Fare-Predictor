# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy Steps

### 1. Go to Streamlit Cloud
Visit: https://share.streamlit.io/

### 2. Sign in with GitHub
- Click "Sign in with GitHub"
- Authorize Streamlit Cloud to access your repositories

### 3. Deploy Your App
1. Click "New app" button
2. Fill in the deployment form:
   - **Repository**: `ajbenc/Trip-And-Fare-Predictor`
   - **Branch**: `master`
   - **Main file path**: `src/interface/web/streamlit_app.py`
   - **App URL** (optional): Choose a custom subdomain like `nyc-taxi-predictor`

3. Click "Deploy!"

### 4. Wait for Deployment
- Streamlit Cloud will install dependencies from `requirements.txt` and `packages.txt`
- Initial deployment takes 5-10 minutes
- You'll see build logs in real-time

### 5. Your App is Live! 🎉
Once deployed, your app will be available at:
`https://[your-subdomain].streamlit.app`

## App Features
Your deployed app will have:
- ✅ Interactive map for zone selection
- ✅ Real-time weather integration
- ✅ Route visualization
- ✅ Fare and duration predictions
- ✅ Custom dark theme

## Troubleshooting

### If deployment fails:
1. Check the logs in Streamlit Cloud dashboard
2. Verify all required files are committed:
   - `requirements.txt` ✅
   - `packages.txt` ✅
   - `.streamlit/config.toml` ✅
   - Model files in `models/lightgbm_80_20_full_year/` ✅
   - Data files in `Data/zones/` ✅

### Common Issues:
- **Memory limit**: Streamlit Cloud free tier has 1GB RAM limit
  - Your model files (~100MB) should fit fine
  - If issues occur, consider optimizing model size
  
- **Build timeout**: First build may timeout
  - Simply retry the deployment
  - Subsequent builds are cached and faster

## Sharing Your App
Once live, share your portfolio project:
```
🚕 NYC Taxi Trip Predictor
Live Demo: https://[your-app].streamlit.app
GitHub: https://github.com/ajbenc/Trip-And-Fare-Predictor
```

## Managing Your App
- **Reboot**: Settings → Reboot app
- **View logs**: Manage app → View logs
- **Update**: Push to GitHub master branch → Auto-redeploys
- **Secrets**: Settings → Secrets (for API keys if needed)

## Auto-Deploy on Push
Your app will automatically redeploy when you push changes to the master branch! 🔄
