"""
Requeue tasks stuck in Redis processing queue back to main queue.

Use this after an unexpected worker crash.
"""

import argparse

from app.config import settings
from app.queue.service import get_queue_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover stale tasks from Redis processing queue")
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=max(1, settings.WORKER_STUCK_TIMEOUT_SECONDS // 60),
        help="Only recover tasks older than this timeout",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=100,
        help="Maximum tasks to process in one run",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=settings.WORKER_MAX_RETRIES,
        help="Tasks exceeding this retry_count are moved to dead-letter queue",
    )
    args = parser.parse_args()

    queue_service = get_queue_service()
    recovery = queue_service.requeue_processing_tasks(
        max_tasks=args.max_tasks,
        stuck_after_seconds=args.timeout_minutes * 60,
        max_retries=args.max_retries,
    )
    print(
        "Recovery complete: "
        f"scanned={recovery['scanned']}, "
        f"requeued={recovery['requeued']}, "
        f"dead_lettered={recovery['dead_lettered']}, "
        f"skipped={recovery['skipped']}"
    )


if __name__ == "__main__":
    main()
