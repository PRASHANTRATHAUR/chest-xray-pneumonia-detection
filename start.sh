#!/bin/bash
# Start both API and Frontend

echo "🚀 Starting Chest X-Ray Detection App..."

# Start FastAPI in background
echo "Starting FastAPI..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for API to start
sleep 5

# Start Streamlit
echo "Starting Streamlit..."
streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true

echo "✅ App running!"
echo "API:      http://localhost:8000"
echo "Frontend: http://localhost:8501"