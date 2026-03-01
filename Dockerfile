FROM python:3.10-slim

# System dependencies for OpenCV headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY dlcv_p2_preprocessing_server.py .
COPY dlcv_p2_preprocessor.py .
COPY dlcv_p2_preprocessing_config.py .
COPY dlcv_p2_config.py .
COPY dlcv_p2_preprocessing_ui.html .

# Copy static assets
COPY static/ static/

# Create mount points for data volumes
RUN mkdir -p chest_xray checkpoints results/sweeps

EXPOSE 8000

CMD ["python", "dlcv_p2_preprocessing_server.py"]
