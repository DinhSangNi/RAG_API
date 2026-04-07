import redis
import json

# Check if edit task is still in queue
client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
queue_len = client.LLEN('rag:edit:queue')
print(f'Edit queue length: {queue_len}')

if queue_len > 0:
    # Peek at the task
    task_data = client.LRANGE('rag:edit:queue', 0, 0)
    if task_data:
        task = json.loads(task_data[0])
        print(f'Task ID: {task.get("task_id")}')
        print(f'Document ID: {task.get("document_id")}')
else:
    print('Edit queue is empty - task was processed!')
