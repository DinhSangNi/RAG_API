import requests

# Test a query that will trigger fallback
response = requests.post(
    'http://localhost:8000/api/v1/chat',
    json={'question': 'Chi tiết về cơ cấu tổ chức?'},
    timeout=60
)

if response.status_code == 200:
    data = response.json()
    print('✓ Response received')
    print(f'Mode: {data.get("metadata", {}).get("retrieval_mode")}')
else:
    print(f'Error: {response.status_code}')
