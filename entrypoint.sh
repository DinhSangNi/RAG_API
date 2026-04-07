#!/bin/bash
set -e

echo "=================================================="
echo "🚀 RAG API - All-in-One Container Startup"
echo "=================================================="

# Create data directory
mkdir -p /app/data/temp_uploads

# ========================================
# 1. Start Redis with memory limit
# ========================================
echo ""
echo "📦 Starting Redis Server..."
# Note: Memory overcommit warning can be suppressed in production
# For Azure Container Instances, this warning is expected and safe to ignore
redis-server --port 6379 \
    --maxmemory 256mb \
    --maxmemory-policy allkeys-lru \
    --save 900 1 \
    --logfile /proc/self/fd/1 \
    --daemonize no \
    --stop-writes-on-bgsave-error no \
    &
REDIS_PID=$!
echo "✅ Redis started (PID: $REDIS_PID)"

# Wait for Redis to be ready
sleep 2
echo "⏳ Waiting for Redis to be ready..."
for i in {1..30}; do
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Redis failed to start"
        exit 1
    fi
    sleep 1
done

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
