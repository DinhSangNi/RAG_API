"""
Redis Queue Service for background task processing
"""
import redis
import json
from typing import Dict, Any, Optional
from app.config import settings


class RedisQueueService:
    """Service for managing Redis tasks"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        self.upload_queue = "rag:upload:queue"
        self.edit_queue = "rag:edit:queue"
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
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB
    )
