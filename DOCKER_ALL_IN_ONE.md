# RAG API - All-in-One Docker Container

## Architecture

This version uses a **single Docker container** running all services:

- **FastAPI** (Port 8000): HTTP API server
- **Redis** (Port 6379): Task queue with 256MB memory limit
- **Worker**: Background process for document processing

All services are managed by a shell script (`entrypoint.sh`) using background processes.

## Why All-in-One?

✅ **Simplifies deployment**: Single container on Azure Container Instances
✅ **Avoids coordination issues**: No need to coordinate starting multiple containers
✅ **Reduced resource overhead**: Shared libraries and processes
✅ **Better for small-medium workloads**: 256MB Redis limit prevents abuse
✅ **Easier debugging**: All logs in single container

## Docker Run

```bash
# Build locally
docker build -t rag-api:latest .

# Run with Docker
docker run -p 8000:8000 -p 6379:6379 \
  -e DATABASE_URL="postgresql://..." \
  -e GEMINI_API_KEY="..." \
  -e CLOUDINARY_CLOUD_NAME="..." \
  -e CLOUDINARY_API_KEY="..." \
  -e CLOUDINARY_API_SECRET="..." \
  rag-api:latest

# Run with Docker Compose
docker-compose up
```

## Ports Exposed

| Port | Service | Purpose |
|------|---------|---------|
| 8000 | FastAPI | HTTP API |
| 6379 | Redis | Task Queue |

## Container Startup Sequence

1. **Redis starts** with:
   - Max memory: 256MB
   - LRU eviction policy (least recently used items removed when full)
   - Persistence: Every 900 seconds with 1 change
   
2. **Worker starts** in background
   - Polls Redis queue every 1 second
   - Processes upload/edit tasks asynchronously
   - Stores results in Redis

3. **FastAPI starts** (foreground)
   - Listens on `0.0.0.0:8000`
   - Handles HTTP requests
   - Pushes tasks to Redis queue

## Configuration

### Default Redis Memory Limits
- **256MB max**: Suitable for production Azure Container Instances
- LRU eviction: Old data automatically removed when limit reached
- Logs: Output to stdout for Azure monitoring

### Adjusting Redis Memory

Edit `entrypoint.sh` line:
```bash
redis-server --port 6379 \
    --maxmemory 256mb \        # ← Adjust here
    --maxmemory-policy allkeys-lru \
    ...
```

Options:
- `256mb`: Production (default, recommended for Azure)
- `512mb`: For larger workloads
- `128mb`: For minimal deployments

## Monitoring

### Check API Health
```bash
curl http://localhost:8000/health
```

### Check Redis
```bash
# From container
redis-cli ping
redis-cli info

# From host (if exposed)
redis-cli -h localhost ping
```

### View Logs
```bash
docker logs -f rag_api_complete
```

## Graceful Shutdown

The container handles signals properly:
- Pressing `Ctrl+C` gracefully shuts down all services
- Redis saves final snapshot before exiting
- Worker completes in-progress tasks when possible

## Performance Notes

- **All-in-one**: Best for up to ~100 concurrent document processes
- **Memory**: 256MB Redis + Python process (typically 300-500MB total)
- **CPU**: Depends on embedding/chunking (usually 1-2 cores sufficient)

## Migration to Scaled Setup

If you need to scale beyond single container capabilities:

1. Extract Redis to separate container with larger memory
2. Extract Worker to separate container with more CPU
3. Run multiple Worker replicas behind a load balancer
4. See legacy `docker-compose.yml` pattern for reference

This all-in-one setup is optimized for Azure Container Instances deployment.
