# CIRCUITRON — Deployment Guide

## Architecture

| Layer    | Service | What it runs |
|----------|---------|--------------|
| Frontend | **Vercel** (free tier) | Next.js 14 app (`frontend/`) |
| Backend  | **Render** (starter or free tier) | FastAPI + YOLO + CRNN OCR (`test/`) |

TrOCR (the "slow" 1.2 GB model) is **excluded** from deployment. Only the fast CRNN OCR (32 MB) is used in production.

---

## Prerequisites

1. Push your repo to GitHub (it's already at `git@github.com:utsabfdahal/FinalCodeofCircuitron.git`)
2. Ensure the CRNN model and CircuitJS files are committed (see "Before You Push" below)
3. Accounts on [Vercel](https://vercel.com) and [Render](https://render.com)

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

## Step 1: Deploy Backend on Render

### Option A: One-click (render.yaml)

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New** → **Blueprint**
3. Connect your GitHub repo
4. Render reads `render.yaml` and auto-configures the service
5. Set the `LIGHTNING_AI_API_KEY` env var in the dashboard

### Option B: Manual setup

1. Click **New** → **Web Service**
2. Connect your GitHub repo
3. Configure:
   - **Name**: `circuitron-api`
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements-deploy.txt`
   - **Start Command**: `uvicorn test.main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment**:
   - `PYTHON_VERSION` = `3.11`
   - `LIGHTNING_AI_API_KEY` = your key from `.env`
   - `ALLOWED_ORIGINS` = `https://your-app.vercel.app` (set after Vercel deploy)
5. Click **Deploy**

### Important Notes

- **Plan**: The free tier has 512 MB RAM. YOLOv7 + CRNN needs ~1–2 GB. Use the **Starter plan** ($7/mo) or higher.
- **Disk**: Model files are committed to git so Render gets them automatically. No persistent disk needed.
- **Cold starts**: Free tier spins down after inactivity. First request takes ~30–60s to load models.
- **PyTorch CPU**: `requirements-deploy.txt` pins `torch` CPU-only to keep the image smaller (~200 MB vs 2 GB with CUDA).

### Verify Backend

Once deployed, visit: `https://circuitron-api.onrender.com/docs` — you should see the FastAPI Swagger UI.

---

## Step 2: Deploy Frontend on Vercel

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **Add New** → **Project**
3. Import your GitHub repo
4. Configure:
   - **Framework Preset**: Next.js (auto-detected)
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `.next` (default)
5. Under **Environment Variables**, add:
   ```
   NEXT_PUBLIC_API_URL = https://circuitron-api.onrender.com
   ```
   (Replace with your actual Render URL)
6. Click **Deploy**

### After Vercel Deploy

Copy your Vercel URL (e.g. `https://circuitron.vercel.app`) and go back to Render:
- Update `ALLOWED_ORIGINS` env var to: `https://circuitron.vercel.app`

---

## Step 3: Verify End-to-End

1. Open your Vercel URL
2. Upload a circuit image
3. The frontend calls `NEXT_PUBLIC_API_URL/analyze` → Render backend
4. Backend runs YOLO detection + CRNN OCR + line detection
5. Results appear in the review step

---

## Environment Variables Summary

### Backend (Render)
| Variable | Value | Required |
|----------|-------|----------|
| `PYTHON_VERSION` | `3.11` | Yes |
| `LIGHTNING_AI_API_KEY` | Your Lightning AI key | Yes (for chat) |
| `ALLOWED_ORIGINS` | `https://your-app.vercel.app` | Recommended |

### Frontend (Vercel)
| Variable | Value | Required |
|----------|-------|----------|
| `NEXT_PUBLIC_API_URL` | `https://circuitron-api.onrender.com` | Yes |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: transformers` | Expected — TrOCR is excluded. Ensure `ocr_mode=fast` in frontend. |
| CORS errors in browser console | Set `ALLOWED_ORIGINS` on Render to your Vercel domain |
| Backend OOM / crashes | Upgrade Render plan (YOLOv7 needs ~1 GB RAM) |
| Models not found | Ensure `yolov7new/best.pt` and `customOCR/crnn_last (1).pth` are committed |
| CircuitJS simulation page blank | Ensure `frontend/public/circuitjs/` is committed |
| Slow first request | Render free tier cold-starts; upgrade to Starter to keep alive |

---

## Optional: Custom Domain

- **Vercel**: Settings → Domains → Add your domain
- **Render**: Settings → Custom Domains → Add domain
- Update `NEXT_PUBLIC_API_URL` and `ALLOWED_ORIGINS` accordingly
