# CIRCUITRON — Deployment Guide

## Architecture

| Layer    | Service | What it runs |
|----------|---------|--------------|
| Frontend | **Vercel** (free tier) | Next.js 14 app (`frontend/`) |
| Backend  | **Railway** (hobby plan) | FastAPI + YOLO + CRNN OCR (`test/`) |

TrOCR (the "slow" 1.2 GB model) is **excluded** from deployment. Only the fast CRNN OCR (32 MB) is used in production.

---

## Prerequisites

1. Push your repo to GitHub (`git@github.com:utsabfdahal/HostingCircuitron.git`)
2. Ensure the CRNN model and CircuitJS files are committed (see "Before You Push" below)
3. Accounts on [Vercel](https://vercel.com) and [Railway](https://railway.app)

---

## Before You Push

The `.gitignore` has been updated to un-ignore files needed for deployment. Run:

```bash
# Add the CRNN model weight (32 MB)
git add -f "customOCR/crnn_last (1).pth"

# Add CircuitJS static files for the simulation page (7.5 MB)
git add frontend/public/circuitjs/

# Commit
git commit -m "chore: add deployment assets (CRNN model + CircuitJS static files)"
git push origin main
```

---

## Step 1: Deploy Backend on Railway

### 1.1 Create a Railway Account

1. Go to [railway.app](https://railway.app)
2. Click **"Login"** → Sign in with **GitHub**

### 1.2 Create a New Project

1. Click **"New Project"**
2. Select **"Deploy from GitHub Repo"**
3. Find and select **`utsabfdahal/HostingCircuitron`**
4. Click **"Deploy Now"**

### 1.3 Configure Environment Variables

Railway will start building immediately. While it builds:

1. Click on your service (the purple box)
2. Go to the **"Variables"** tab
3. Add these variables:

| Variable | Value |
|----------|-------|
| `LIGHTNING_AI_API_KEY` | Your Lightning AI key from `.env` |
| `ALLOWED_ORIGINS` | `*` (update to your Vercel domain later) |

### 1.4 Set the Install Command

1. Go to the **"Settings"** tab on your service
2. Under **Build**, set **Install Command** to:
   ```
   pip install -r requirements-deploy.txt
   ```
   This uses the slimmed-down deps without TrOCR — faster builds, smaller image.

### 1.5 Wait for Deployment

- Watch the **"Deployments"** tab for build logs
- Once it says **"Active"**, your API is live

### 1.6 Get Your Public URL

1. Go to **"Settings"** tab
2. Under **"Networking"** → **"Public Networking"**
3. Click **"Generate Domain"**
4. Railway gives you a URL like:
   ```
   https://hostingcircuitron-production.up.railway.app
   ```

### 1.7 Verify Backend

Open in your browser:
```
https://hostingcircuitron-production.up.railway.app/docs
```
You should see the FastAPI Swagger UI.

### Railway Plan Info

| Feature | Detail |
|---------|--------|
| **Trial** | $5 free credit, no credit card needed |
| **Hobby plan** | $5/month + usage-based |
| **RAM** | Up to 8 GB available |
| **Cold starts** | None on Hobby plan |
| **Auto-deploy** | On every `git push` |
| **PyTorch** | CPU-only via `requirements-deploy.txt` (~200 MB vs 2 GB CUDA) |

---

## Step 2: Deploy Frontend on Vercel

### 2.1 Create the Project

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **"Add New"** → **"Project"**
3. Import your GitHub repo (`HostingCircuitron`)

### 2.2 Configure Build Settings

| Setting | Value |
|---------|-------|
| **Framework Preset** | Next.js (auto-detected) |
| **Root Directory** | `frontend` |
| **Build Command** | `npm run build` (default) |
| **Output Directory** | `.next` (default) |

### 2.3 Set Environment Variables

Add this under **Environment Variables**:

```
NEXT_PUBLIC_API_URL = https://hostingcircuitron-production.up.railway.app
```
*(Replace with your actual Railway URL from Step 1.6)*

### 2.4 Deploy

Click **"Deploy"** and wait for the build to complete.

### 2.5 Link CORS

After Vercel deploys, copy your Vercel URL (e.g. `https://circuitron.vercel.app`) and go back to Railway:

1. Open your service → **"Variables"** tab
2. Update `ALLOWED_ORIGINS` to your Vercel domain:
   ```
   ALLOWED_ORIGINS = https://circuitron.vercel.app
   ```

---

## Step 3: Verify End-to-End

1. Open your Vercel URL
2. Upload a circuit image
3. The frontend calls `NEXT_PUBLIC_API_URL/analyze` → Railway backend
4. Backend runs YOLO detection + CRNN OCR + line detection
5. Results appear in the review step

---

## Environment Variables Summary

### Backend (Railway)

| Variable | Value | Required |
|----------|-------|----------|
| `LIGHTNING_AI_API_KEY` | Your Lightning AI key | Yes (for chat) |
| `ALLOWED_ORIGINS` | `https://your-app.vercel.app` | Recommended |

### Frontend (Vercel)

| Variable | Value | Required |
|----------|-------|----------|
| `NEXT_PUBLIC_API_URL` | `https://your-app.up.railway.app` | Yes |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: transformers` | Expected — TrOCR is excluded. Ensure `ocr_mode=fast` in frontend. |
| CORS errors in browser console | Set `ALLOWED_ORIGINS` on Railway to your Vercel domain |
| Backend OOM / crashes | Railway supports up to 8 GB — should not happen |
| Models not found | Ensure `yolov7new/best.pt` and `customOCR/crnn_last (1).pth` are committed |
| CircuitJS simulation page blank | Ensure `frontend/public/circuitjs/` is committed |
| Build too slow | `requirements-deploy.txt` uses CPU-only PyTorch to speed up builds |

---

## Optional: Custom Domain

- **Vercel**: Settings → Domains → Add your domain
- **Railway**: Settings → Custom Domains → Add domain
- Update `NEXT_PUBLIC_API_URL` and `ALLOWED_ORIGINS` accordingly
