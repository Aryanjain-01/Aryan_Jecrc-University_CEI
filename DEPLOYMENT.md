# 🚀 Deploy SimilarityX AI to Render

Complete guide to deploy your SimilarityX AI visual product recommendation engine to Render.

## ✅ What We've Set Up

Your repository now has all necessary deployment files:

- ✅ `render.yaml` — Render deployment configuration
- ✅ `requirements.txt` — Python dependencies
- ✅ `.streamlit/config.toml` — Streamlit UI settings
- ✅ `DEPLOYMENT.md` — This guide

## 🎯 Quick Start (3 Steps)

### Step 1: Go to Render Dashboard
Visit [https://dashboard.render.com](https://dashboard.render.com)

### Step 2: Create Web Service
1. Click **"New +"** button (top-right)
2. Select **"Web Service"**
3. Click **"Connect your GitHub account"** (if not already connected)
4. Search for and select: `Aryan_Jecrc-University_CEI`
5. Click **"Connect"**

### Step 3: Deploy
1. Render will auto-detect `render.yaml`
2. Review default settings:
   - **Name:** `similarityx-recommender`
   - **Environment:** Python
   - **Plan:** Free
3. Click **"Create Web Service"**
4. Wait 3-5 minutes for build to complete ⏳

## 🌐 Your Live App

Once deployed, your app will be live at:
```
https://similarityx-recommender.onrender.com
```

## 📋 Deployment Configuration

**render.yaml summary:**
- Installs dependencies from `requirements.txt`
- Runs Streamlit app from: `Similarityx- Visual Product Recommender [WEEK 8]/app/streamlit_app.py`
- Uses port `$PORT` (assigned by Render)
- Logs errors for debugging

## ⚠️ Important: Data & Models

Your app needs these files to work:

### Required:
1. **Catalog CSV:** `Similarityx- Visual Product Recommender [WEEK 8]/data/subset/subset.csv`
2. **Product Images:** `Similarityx- Visual Product Recommender [WEEK 8]/data/subset/images/`
3. **Pre-computed Embeddings:** `Similarityx- Visual Product Recommender [WEEK 8]/artifacts/embeddings/`

### First Time Setup:

If models/data are NOT in your repo yet, run locally first:

```bash
# 1. Navigate to project directory
cd "Similarityx- Visual Product Recommender [WEEK 8]"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate embeddings
python -m src.embeddings

# 4. Commit and push to GitHub
cd ../..
git add "Similarityx- Visual Product Recommender [WEEK 8]/artifacts/embeddings/"
git add "Similarityx- Visual Product Recommender [WEEK 8]/data/subset/"
git commit -m "Add pre-computed embeddings and dataset subset for deployment"
git push
```

Then redeploy on Render (or Render will auto-redeploy on push).

## 🧪 Testing Your Deployment

1. **Wait for build to complete** — Check the Render dashboard logs
2. **Visit your app URL** — `https://similarityx-recommender.onrender.com`
3. **Click "Surprise me"** — Test with a random catalog item
4. **Upload an image** — Test with your own fashion image

## 🐛 Troubleshooting

### App shows blank page
- Check Render logs: Dashboard → Your Service → Logs
- Look for Python errors
- Ensure all required files are committed to git

### "Model not found" error
- Models are missing from git
- Follow "First Time Setup" above
- Generate embeddings locally and commit

### Slow first request (~30s)
- Normal behavior! Model is loading on first run
- Subsequent requests are instant
- Render Free tier may have slower cold starts

### Build fails
- Check build logs for dependency errors
- Verify `requirements.txt` is valid
- Try rebuilding from Render dashboard

## 📊 Live Features

Once deployed, your app has:

✅ **Upload custom images** for similarity search  
✅ **"Surprise me"** button for random catalog items  
✅ **Filter by:** Category, Gender, Color  
✅ **Compare models:** Baseline vs Improved  
✅ **Real-time results** with similarity scores  
✅ **Instant subsequent searches** (cached model)  

## 💡 Free Tier Limits

- **Compute:** 0.5 CPU, 512 MB RAM
- **Bandwidth:** 100 GB/month
- **Inactivity:** Service suspends after 15 mins without traffic
- **Restart:** First request after suspension takes ~30s

For production, upgrade to Paid tier on Render.

## 🔄 Redeploying

To redeploy after code changes:

**Option 1: Auto-deploy (Recommended)**
```bash
git add .
git commit -m "Your changes"
git push
# Render auto-redeploys on push
```

**Option 2: Manual redeploy**
- Render Dashboard → Your Service → Manual Deploy

## 📞 Support

- **Render Docs:** https://render.com/docs
- **Streamlit Docs:** https://docs.streamlit.io
- **GitHub Issues:** Check your repo's issues tab

---

**Happy deploying! 🎉**

Questions? Check the logs first—they're your best friend!
