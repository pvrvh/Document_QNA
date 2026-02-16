# Railway Deployment Guide

## 🚂 Deploy Your RAG App to Railway

### Step 1: Sign Up / Login to Railway
1. Go to https://railway.app
2. Click "Login with GitHub"
3. Authorize Railway to access your GitHub

### Step 2: Create New Project
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your repository: **pvrvh/Document_QNA**
4. Railway will auto-detect it's a Python app

### Step 3: Add Environment Variable
**IMPORTANT:** Add your Groq API key
1. Go to your project → Variables tab
2. Click "New Variable"
3. Add:
   - **Key**: `GROQ_API_KEY`
   - **Value**: `your-groq-api-key-here` (get it from https://console.groq.com/keys)
4. Click "Add"

### Step 4: Deploy
1. Railway will automatically start building
2. Wait 5-10 minutes for:
   - Python dependencies to install
   - Embedding model to download (~90MB)
   - Server to start

### Step 5: Get Your URL
1. Go to Settings tab
2. Click "Generate Domain"
3. Your app will be live at: `your-app.railway.app`

### Step 6: Test
Open your Railway URL and:
- Upload a document
- Ask a question
- See the streaming AI answer! 🤖

## 📊 Features on Railway
- ✅ Persistent storage (ChromaDB works!)
- ✅ Always-on container
- ✅ Auto-deploys on git push
- ✅ Free $5/month credit (enough for hobby projects)
- ✅ Custom domains (optional)

## 🔧 Troubleshooting
If deployment fails:
- Check Logs tab in Railway
- Verify GROQ_API_KEY is set
- Ensure PORT is not hardcoded (Railway sets it automatically)

---

**Your GitHub repo:** https://github.com/pvrvh/Document_QNA
**Ready to deploy!** 🚀
