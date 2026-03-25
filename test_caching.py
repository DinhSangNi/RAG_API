import requests

for i in range(2):
    print(f'\n=== Request {i+1} ===')
    try:
        response = requests.post(
            'http://localhost:8000/api/v1/chat',
            json={'question': 'Hồ Chí Minh là ai?'},
            timeout=45
        )
        if response.status_code == 200:
            data = response.json()
            timing = data.get('metadata', {}).get('timing', {})
            print(f'Success - Retrieval: {timing.get("retrieval_s")}s, Generation: {timing.get("generation_s")}s, Total: {timing.get("total_s")}s')
        else:
            print(f'Error {response.status_code}')
    except Exception as e:
        print(f'Exception: {e}')
