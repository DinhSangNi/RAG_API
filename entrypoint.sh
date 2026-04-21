#!/bin/bash
set -e

echo "=================================================="
echo "🚀 RAG API - All-in-One Container Startup"
echo "=================================================="

# Create data directory
mkdir -p /app/data/temp_uploads

# ========================================
# 1. Redis is external - configured via REDIS_CONNECTION_STRING
# ========================================
echo ""
echo "🌐 Using external Redis from environment"
if [ -z "$REDIS_CONNECTION_STRING" ]; then
    echo "⚠️  REDIS_CONNECTION_STRING is not set - using REDIS_HOST:REDIS_PORT fallback"
else
    echo "✅ REDIS_CONNECTION_STRING is configured"
fi

# ========================================
# 2. Start Worker in background
# ========================================
echo ""
echo "👷 Starting Background Worker..."
cd /app
python worker.py > /proc/self/fd/1 2>&1 &
WORKER_PID=$!
echo "✅ Worker started (PID: $WORKER_PID)"

# ========================================
# 3. Start FastAPI (foreground)
# ========================================
echo ""
echo "🌐 Starting FastAPI Server..."
cd /app
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
