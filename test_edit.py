import requests
import json
import time
import sys

# First get a document
print("📋 Fetching documents...")
response = requests.get('http://localhost:8000/api/v1/documents?limit=1')
if response.status_code != 200:
    print(f"❌ Failed to fetch documents: {response.status_code}")
    sys.exit(1)

docs = response.json()
if not docs:
    print("❌ No documents found. Please upload a document first.")
    sys.exit(1)

document = docs[0]
doc_id = document['id']
print(f"✅ Found document: {doc_id}")
print(f"   Name: {document['file_name']}")
print(f"   Status: {document['status']}")

# Create updated file
print("\n📝 Creating updated test file...")
updated_content = """# Test Document - Updated

This is the updated version of the document.

## New Section 1
Updated content here.

## New Section 2  
More updated content.

## New Section 3
Even more content after editing.
"""

with open('C:\\Temp\\test_doc_updated.md', 'w', encoding='utf-8') as f:
    f.write(updated_content)

print("✅ Updated file created")

# Send edit request
print(f"\n🔄 Sending edit request for document {doc_id}...")
with open('C:\\Temp\\test_doc_updated.md', 'rb') as f:
    files = [('file', (f.name, f, 'text/markdown'))]
    response = requests.put(f'http://localhost:8000/api/v1/documents/{doc_id}/edit', files=files)
    
    if response.status_code != 200:
        print(f"❌ Edit failed: {response.status_code}")
        print(response.json())
        sys.exit(1)
    
    result = response.json()
    print(f"✅ Edit queued successfully")
    print(json.dumps(result, indent=2))

print("\n⏳ Waiting 15 seconds for worker to process edit...")
time.sleep(15)

print("✅ Test completed")
