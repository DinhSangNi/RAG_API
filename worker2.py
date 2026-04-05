#!/usr/bin/env python
"""
RQ Worker 2 - Document Indexing Worker
Processes queued document indexing jobs
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rq import Worker, Queue
from redis import Redis
from app.config import settings

# Ensure services can be imported
from app.workers.queue_job import index_document_job  # noqa: F401


def main():
    """Start RQ Worker 2"""
    print("\n" + "="*70)
    print("🚀 RQ WORKER 2 - STARTING")
    print("="*70)
    print(f"📊 Configuration:")
    print(f"   Redis Host: {settings.REDIS_HOST}")
    print(f"   Redis Port: {settings.REDIS_PORT}")
    print(f"   Redis DB: {settings.REDIS_DB}")
    print(f"   Queue Name: {settings.QUEUE_NAME}")
    print(f"   Database: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else '...'}")
    print("="*70 + "\n")

    try:
        # Connect to Redis
        redis_conn = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )

        # Test Redis connection
        redis_conn.ping()
        print("✅ Connected to Redis\n")

        # Create queue and worker
        queue = Queue(settings.QUEUE_NAME, connection=redis_conn)
        worker = Worker([queue], connection=redis_conn, name="worker-2")

        print("🎯 Listening for jobs...")
        print("⏸️  Press Ctrl+C to stop worker\n")

        # Start worker
        worker.work(with_scheduler=False)

    except ConnectionError as e:
        print(f"❌ Failed to connect to Redis: {e}")
        print("   Make sure Redis is running!")
        print(f"   Redis URL: {settings.REDIS_URL}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Worker stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Worker error: {e}\n")
        raise


if __name__ == "__main__":
    main()
