"""
Redis Queue Service for background task processing
"""
import redis
import json
from typing import Dict, Any, Optional
from urllib.parse import unquote
from app.config import settings


def _create_redis_client(
    connection_string: str = "",
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
) -> redis.Redis:
    normalized_connection_string = connection_string.strip()

    if normalized_connection_string:
        if normalized_connection_string.startswith(("redis://", "rediss://")):
            return redis.Redis.from_url(
                normalized_connection_string,
                decode_responses=True,
            )

        parts = [part.strip() for part in normalized_connection_string.split(",") if part.strip()]
        host_part = parts[0]
        if ":" in host_part:
            redis_host, redis_port = host_part.rsplit(":", 1)
        else:
            redis_host, redis_port = host_part, str(port)

        options: Dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            options[key.strip().lower()] = value.strip()

        use_ssl = options.get("ssl", "false").lower() == "true"
        redis_db = int(options.get("db", str(db)))
        username = options.get("user") or options.get("username")
        password = options.get("password")

        return redis.Redis(
            host=unquote(redis_host),
            port=int(redis_port),
            db=redis_db,
            username=unquote(username) if username else None,
            password=unquote(password) if password else None,
            ssl=use_ssl,
            decode_responses=True,
        )

    return redis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=True,
    )


class RedisQueueService:
    """Service for managing Redis tasks.
    
    Queue names match Wiki Backend (C# StackExchange.Redis):
    - document:task:queue: New tasks from Wiki Backend
    - document:processing:queue: Tasks currently being processed
    - document:failed:queue: Failed tasks
    - rag:result:{task_id}: Task results (stored by RAG_API)
    """
    
    def __init__(
        self,
        connection_string: str = "",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
    ):
        self.redis_client = _create_redis_client(
            connection_string=connection_string,
            host=host,
            port=port,
            db=db,
        )
        # Match Wiki Backend queue names from RedisDocumentQueueService
        self.upload_queue = "document:task:queue"
        self.edit_queue = "document:task:queue"  # Uses same queue; type determined by JSON
        self.processing_queue = "document:processing:queue"
        self.failed_queue = "document:failed:queue"
        self.result_prefix = "rag:result:"
    
    def push_upload_task(self, task_data: Dict[str, Any]) -> str:
        """Push upload task to queue"""
        task_id = task_data.get("task_id")
        self.redis_client.rpush(self.upload_queue, json.dumps(task_data))
        print(f"✅ Task pushed to upload queue: {task_id}")
        return task_id
    
    def push_edit_task(self, task_data: Dict[str, Any]) -> str:
        """Push edit task to queue"""
        task_id = task_data.get("task_id")
        self.redis_client.rpush(self.edit_queue, json.dumps(task_data))
        print(f"✅ Task pushed to edit queue: {task_id}")
        return task_id
    
    def pop_upload_task(self) -> Optional[Dict[str, Any]]:
        """Pop job from upload queue (non-blocking)"""
        result = self.redis_client.lpop(self.upload_queue)
        if result:
            return json.loads(result)
        return None
    
    def pop_edit_task(self) -> Optional[Dict[str, Any]]:
        """Pop job from edit queue (non-blocking)"""
        result = self.redis_client.lpop(self.edit_queue)
        if result:
            return json.loads(result)
        return None
    
    def set_result(self, task_id: str, result_data: Dict[str, Any]) -> None:
        """Store task result with TTL (24 hours)"""
        key = f"{self.result_prefix}{task_id}"
        self.redis_client.setex(
            key,
            86400,  # 24 hours
            json.dumps(result_data)
        )
        print(f"✅ Result stored for task: {task_id}")
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result"""
        key = f"{self.result_prefix}{task_id}"
        result = self.redis_client.get(key)
        if result:
            return json.loads(result)
        return None
    
    def health_check(self) -> bool:
        """Check Redis connection"""
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return False


def get_queue_service() -> RedisQueueService:
    """Get or create queue service instance from settings"""
    return RedisQueueService(
        connection_string=settings.REDIS_CONNECTION_STRING,
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB
    )
