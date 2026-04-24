"""
Redis Queue Service for background task processing
"""
import os
import redis
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from urllib.parse import unquote
from app.config import settings


TASK_STATUS_PENDING = "PENDING"
TASK_STATUS_PROCESSING = "PROCESSING"
TASK_STATUS_COMPLETED = "COMPLETED"
TASK_STATUS_FAILED = "FAILED"
MAX_RETRIES_DEFAULT = 3


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

    Queue names (shared between RAG + Graph-RAG workers):
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
        self.main_queue = os.getenv("REDIS_TASK_QUEUE", "document:task:queue:rag")
        self.upload_queue = self.main_queue
        self.edit_queue = self.main_queue  # Uses same queue; type determined by JSON
        self.processing_queue = os.getenv("REDIS_PROCESSING_QUEUE", "document:processing:queue:rag")
        self.failed_queue = os.getenv("REDIS_FAILED_QUEUE", "document:failed:queue:rag")
        self.dead_letter_queue = os.getenv("REDIS_DEAD_LETTER_QUEUE", "document:dead-letter:queue")
        self.result_prefix = os.getenv("WORKER_RESULT_PREFIX", "rag:result:")
        self.status_prefix = os.getenv("WORKER_STATUS_PREFIX", "rag:task:status:")
        self._sentinel_cache: Dict[str, str] = {}
        self._ensure_managed_queues_exist()

    def _sentinel_payload(self, queue_name: str) -> str:
        cached = self._sentinel_cache.get(queue_name)
        if cached:
            return cached

        payload = json.dumps(
            {
                "sentinel": True,
                "type": "sentinel",
                "queue": queue_name,
            },
            separators=(",", ":"),
        )
        self._sentinel_cache[queue_name] = payload
        return payload

    def _is_sentinel_payload(self, raw_payload: Any, queue_name: Optional[str] = None) -> bool:
        if not isinstance(raw_payload, str):
            return False

        try:
            payload = json.loads(raw_payload)
        except Exception:
            return False

        if payload.get("sentinel") is not True and payload.get("type") != "sentinel":
            return False

        if queue_name is None:
            return True

        return payload.get("queue") == queue_name

    def _ensure_queue_exists(self, queue_name: str) -> None:
        if self.redis_client.exists(queue_name):
            return
        self.redis_client.rpush(queue_name, self._sentinel_payload(queue_name))

    def _ensure_managed_queues_exist(self) -> None:
        for queue_name in (
            self.main_queue,
            self.processing_queue,
            self.failed_queue,
            self.dead_letter_queue,
        ):
            self._ensure_queue_exists(queue_name)

    def _remove_queue_sentinel(self, queue_name: str) -> None:
        self.redis_client.lrem(queue_name, 0, self._sentinel_payload(queue_name))

    def _restore_queue_sentinel_if_empty(self, queue_name: str) -> None:
        if self.redis_client.llen(queue_name) == 0:
            self.redis_client.rpush(queue_name, self._sentinel_payload(queue_name))

    def _push_queue_item(self, queue_name: str, payload: Dict[str, Any]) -> None:
        self._remove_queue_sentinel(queue_name)
        self.redis_client.rpush(queue_name, json.dumps(payload))

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_iso_datetime(value: Any) -> Optional[datetime]:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None

    @staticmethod
    def _normalize_pipeline_type(value: Any) -> Optional[str]:
        normalized = str(value or "").strip().lower()
        if normalized in {"rag", "rag-api", "rag_api", "ragworker", "rag-worker"}:
            return "rag"
        if normalized in {"graph", "graph-rag", "graph_rag", "graphrag", "graph-worker", "graph_worker"}:
            return "graph"
        return None

    def _is_task_stuck(
        self,
        task_status: Optional[Dict[str, Any]],
        now: datetime,
        stuck_after_seconds: int,
    ) -> bool:
        if not task_status:
            return True

        processing_started_at = self._parse_iso_datetime(task_status.get("processing_started_at"))
        updated_at = self._parse_iso_datetime(task_status.get("updated_at"))

        # Only recover tasks that have been in PROCESSING long enough.
        if str(task_status.get("status", "")).upper() != TASK_STATUS_PROCESSING:
            return False

        if processing_started_at and (now - processing_started_at).total_seconds() < stuck_after_seconds:
            return False

        # Guard against requeueing healthy active workers that are still heartbeat-updating.
        if updated_at and (now - updated_at).total_seconds() < stuck_after_seconds:
            return False

        return True
    
    def push_upload_task(self, task_data: Dict[str, Any]) -> str:
        """Push upload task to queue"""
        task_data = dict(task_data)
        task_data["type"] = "rag"
        task_data["pipeline"] = "rag"
        task_data["task_type"] = "upload"
        task_data["retry_count"] = self._to_int(task_data.get("retry_count"), 0)
        task_id = str(task_data.get("task_id"))
        self._push_queue_item(self.main_queue, task_data)
        self.set_task_status(
            task_id=task_id,
            status=TASK_STATUS_PENDING,
            message="Task queued and waiting for worker",
            task_type="upload",
            file_name=task_data.get("file_name"),
            retry_count=task_data.get("retry_count", 0),
        )
        print(f"✅ Task pushed to upload queue: {task_id}")
        return task_id
    
    def push_edit_task(self, task_data: Dict[str, Any]) -> str:
        """Push edit task to queue"""
        task_data = dict(task_data)
        task_data["type"] = "rag"
        task_data["pipeline"] = "rag"
        task_data["task_type"] = "edit"
        task_data["retry_count"] = self._to_int(task_data.get("retry_count"), 0)
        task_id = str(task_data.get("task_id"))
        self._push_queue_item(self.main_queue, task_data)
        self.set_task_status(
            task_id=task_id,
            status=TASK_STATUS_PENDING,
            message="Task queued and waiting for worker",
            task_type="edit",
            file_name=task_data.get("file_name"),
            document_id=task_data.get("document_id"),
            retry_count=task_data.get("retry_count", 0),
        )
        print(f"✅ Task pushed to edit queue: {task_id}")
        return task_id

    def claim_task_blocking(self, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """
        Atomically move one task from main queue to processing queue.

        Uses BLMOVE LEFT->RIGHT so queue consumption is FIFO while preserving
        atomic handoff to processing queue.

        If Redis does not support BLMOVE, falls back to BRPOPLPUSH.
        In that fallback mode the queue behaves like LIFO.

        Regardless of command, if the worker crashes mid-processing, the task
        remains in processing_queue for later recovery.
        """
        try:
            raw_payload = self.redis_client.execute_command(
                "BLMOVE",
                self.main_queue,
                self.processing_queue,
                "LEFT",
                "RIGHT",
                timeout,
            )
        except Exception:
            raw_payload = self.redis_client.brpoplpush(
                self.main_queue,
                self.processing_queue,
                timeout=timeout,
            )
        if not raw_payload:
            return None

        self._restore_queue_sentinel_if_empty(self.main_queue)

        # Detect sentinel value used to keep the queue key visible in Redis when
        # the queue is otherwise empty. Put it on the RIGHT so LEFT-pop workers
        # do not claim the sentinel ahead of real tasks.
        try:
            maybe_sentinel = json.loads(raw_payload)
            if maybe_sentinel.get("sentinel") is True or maybe_sentinel.get("type") == "sentinel":
                self.redis_client.lrem(self.processing_queue, 1, raw_payload)
                self.redis_client.rpush(self.main_queue, raw_payload)
                self._restore_queue_sentinel_if_empty(self.processing_queue)
                return None
        except Exception:
            pass

        task_data = json.loads(raw_payload)
        task_data["retry_count"] = self._to_int(task_data.get("retry_count"), 0)
        return {
            "raw_payload": raw_payload,
            "task": task_data,
        }

    def ack_processing_task(self, raw_payload: str) -> int:
        """Acknowledge a processed task by removing it from processing queue."""
        removed = int(self.redis_client.lrem(self.processing_queue, 1, raw_payload))
        self._restore_queue_sentinel_if_empty(self.processing_queue)
        return removed

    def release_unhandled_task(self, raw_payload: str) -> int:
        """Return a claimed task to the main queue when this worker is not the owner."""
        if not raw_payload:
            return 0

        removed = int(self.redis_client.lrem(self.processing_queue, 1, raw_payload))
        if not removed:
            return 0

        self._restore_queue_sentinel_if_empty(self.processing_queue)
        # Push released tasks to the opposite end so this worker does not
        # immediately reclaim the same non-owner payload on the next LEFT-pop.
        self._remove_queue_sentinel(self.main_queue)
        self.redis_client.rpush(self.main_queue, raw_payload)
        return removed
    
    def pop_upload_task(self) -> Optional[Dict[str, Any]]:
        """Pop job from upload queue (non-blocking)"""
        result = self.redis_client.lpop(self.upload_queue)
        if self._is_sentinel_payload(result, self.upload_queue):
            self.redis_client.rpush(self.upload_queue, result)
            return None
        if result:
            self._restore_queue_sentinel_if_empty(self.upload_queue)
            return json.loads(result)
        return None
    
    def pop_edit_task(self) -> Optional[Dict[str, Any]]:
        """Pop job from edit queue (non-blocking)"""
        result = self.redis_client.lpop(self.edit_queue)
        if self._is_sentinel_payload(result, self.edit_queue):
            self.redis_client.rpush(self.edit_queue, result)
            return None
        if result:
            self._restore_queue_sentinel_if_empty(self.edit_queue)
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

    def _status_key(self, task_id: str) -> str:
        return f"{self.status_prefix}{task_id}"

    def set_task_status(
        self,
        task_id: str,
        status: str,
        message: str = "",
        progress: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        file_name: Optional[str] = None,
        document_id: Optional[str] = None,
        task_type: Optional[str] = None,
        ttl_seconds: int = 86400,
        **extra_fields: Any,
    ) -> None:
        """Store task status for frontend polling/observability."""
        key = self._status_key(task_id)
        existing_raw = self.redis_client.get(key)
        existing = json.loads(existing_raw) if existing_raw else {}

        payload = {
            **existing,
            "task_id": task_id,
            "status": status,
            "message": message,
            "updated_at": self._now_iso(),
        }

        if progress is not None:
            payload["progress"] = progress
        if error is not None:
            payload["error"] = error
        if file_name is not None:
            payload["file_name"] = file_name
        if document_id is not None:
            payload["document_id"] = document_id
        if task_type is not None:
            payload["task_type"] = task_type
        if extra_fields:
            payload.update(extra_fields)

        self.redis_client.setex(key, ttl_seconds, json.dumps(payload))

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status object for a task from Redis."""
        key = self._status_key(task_id)
        raw = self.redis_client.get(key)
        if not raw:
            return None
        return json.loads(raw)

    def push_dead_letter_task(self, task_data: Dict[str, Any], reason: str, error: Optional[str] = None) -> None:
        """Move a task payload to dead-letter queue for manual investigation."""
        task_payload = dict(task_data)
        task_payload["dead_lettered_at"] = self._now_iso()
        task_payload["dead_letter_reason"] = reason
        if error:
            task_payload["dead_letter_error"] = error
        self._push_queue_item(self.dead_letter_queue, task_payload)

    def push_failed_task(self, task_data: Dict[str, Any], reason: str, error: Optional[str] = None) -> None:
        """Push a failed task payload to the failed queue for inspection/replay."""
        task_payload = dict(task_data)
        task_payload["failed_at"] = self._now_iso()
        task_payload["failed_reason"] = reason
        if error:
            task_payload["failed_error"] = error
        self._push_queue_item(self.failed_queue, task_payload)

    def requeue_processing_tasks(
        self,
        max_tasks: Optional[int] = None,
        stuck_after_seconds: int = 1800,
        max_retries: int = MAX_RETRIES_DEFAULT,
    ) -> Dict[str, int]:
        """
        Recover stuck tasks from processing queue using timeout and retry safeguards.

        - Requeue tasks only if they are stale (older than stuck_after_seconds)
        - Increment retry_count before requeue
        - Send poison pills to dead_letter_queue when retry_count exceeds max_retries
        """
        now = datetime.now(timezone.utc)
        scanned = 0
        requeued = 0
        dead_lettered = 0
        skipped = 0

        processing_items = self.redis_client.lrange(self.processing_queue, 0, -1)
        for raw_payload in processing_items:
            if max_tasks is not None and (requeued + dead_lettered) >= max_tasks:
                break

            scanned += 1
            if self._is_sentinel_payload(raw_payload, self.processing_queue):
                skipped += 1
                continue
            try:
                task_data = json.loads(raw_payload)
            except json.JSONDecodeError:
                # Corrupted payload should be removed from processing and put in dead-letter.
                removed = self.redis_client.lrem(self.processing_queue, 1, raw_payload)
                if removed:
                    self._restore_queue_sentinel_if_empty(self.processing_queue)
                    self._push_queue_item(
                        self.dead_letter_queue,
                        {
                            "raw_payload": raw_payload,
                            "dead_lettered_at": self._now_iso(),
                            "dead_letter_reason": "Invalid JSON payload in processing queue",
                        },
                    )
                    dead_lettered += 1
                continue

            pipeline_type = self._normalize_pipeline_type(
                task_data.get("type") or task_data.get("pipeline")
            )
            if pipeline_type and pipeline_type != "rag":
                skipped += 1
                continue

            task_id = str(task_data.get("task_id") or "")
            task_status = self.get_task_status(task_id) if task_id else None
            if not self._is_task_stuck(task_status, now, stuck_after_seconds):
                skipped += 1
                continue

            removed = self.redis_client.lrem(self.processing_queue, 1, raw_payload)
            if not removed:
                # Already moved by another admin/worker run.
                skipped += 1
                continue

            self._restore_queue_sentinel_if_empty(self.processing_queue)

            retry_count = self._to_int(task_data.get("retry_count"), 0) + 1
            task_data["retry_count"] = retry_count
            task_data["last_requeued_at"] = self._now_iso()

            if retry_count > max_retries:
                self.push_dead_letter_task(
                    task_data,
                    reason=f"Retry count exceeded max_retries={max_retries}",
                )
                dead_lettered += 1
                if task_id:
                    self.set_task_status(
                        task_id=task_id,
                        status=TASK_STATUS_FAILED,
                        message="Task moved to dead-letter queue after retry limit exceeded",
                        retry_count=retry_count,
                        dead_lettered=True,
                    )
                continue

            self._push_queue_item(self.main_queue, task_data)
            requeued += 1
            if task_id:
                self.set_task_status(
                    task_id=task_id,
                    status=TASK_STATUS_PENDING,
                    message="Task recovered from processing queue and requeued",
                    retry_count=retry_count,
                )

        return {
            "scanned": scanned,
            "requeued": requeued,
            "dead_lettered": dead_lettered,
            "skipped": skipped,
        }
    
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
