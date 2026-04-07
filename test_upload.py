import requests
import json
import time

# Upload file
print("📤 Uploading file...")
with open('C:\\Temp\\test_doc.md', 'rb') as f:
    files = [('files', (f.name, f, 'text/markdown'))]
    data = {'source_type': 'MARKDOWN'}
    response = requests.post('http://localhost:8000/api/v1/upload', files=files, data=data)
    result = response.json()
    print(json.dumps(result, indent=2))
    
    task_id = None
    if isinstance(result, dict) and 'tasks' in result:
        for task in result.get('tasks', []):
            task_id = task.get('task_id')
            print(f'\n✅ Task submitted: {task_id}')

# Wait for processing
if task_id:
    print(f"\n⏳ Waiting 10 seconds for worker to process task...")
    time.sleep(10)
