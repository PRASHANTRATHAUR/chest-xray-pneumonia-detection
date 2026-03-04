# ============================================
# Dockerfile for Chest X-Ray Detection App
# ============================================

# Base image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY api/          ./api/
COPY frontend/     ./frontend/
COPY experiments/  ./experiments/
COPY configs/      ./configs/

# Expose ports
EXPOSE 8000 8501

# Start script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]