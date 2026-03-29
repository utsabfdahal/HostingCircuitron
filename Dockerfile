FROM python:3.11-slim

WORKDIR /app

# System libs needed by opencv-headless, scikit-image, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .

# Step 1: install headless opencv FIRST
RUN pip install --no-cache-dir opencv-python-headless==4.13.0.92

# Step 2: install everything else (ultralytics will try to pull opencv-python GUI, that's ok)
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Step 3: force remove GUI opencv and pin headless
RUN pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless==4.13.0.92

COPY . .

CMD ["sh", "-c", "uvicorn test.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
