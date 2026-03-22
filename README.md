# 🚀 Hierarchical RAG System - Production-Ready Microservice

Hệ thống RAG (Retrieval-Augmented Generation) 4 tầng với kiến trúc phân cấp, được đóng gói thành **Production-Ready Microservice** với FastAPI, PostgreSQL (pgvector + ParadeDB) và Docker.

## 🌟 Highlights

- ✅ **4-Tier Hierarchical RAG** - Kiến trúc phân tầng: Summary Documents → Documents → Parent Chunks → Child Chunks
- ✅ **Intelligent Retrieval** - 6-step workflow với query expansion và scoped search
- ✅ **RESTful API** với FastAPI
- ✅ **Hybrid Search** - BM25 (ParadeDB) + Semantic Search (pgvector)
- ✅ **Duplicate Detection** - SHA256 hash để tránh trùng lặp
- ✅ **Docker Ready** - Deploy một lệnh với docker-compose
- ✅ **Auto Documentation** - Swagger UI tích hợp sẵn

---

## 📊 Kiến trúc 4 tầng

```
summary_documents (Tóm tắt)
    ↓
documents (Tài liệu gốc)
    ↓
parent_chunks (Chunks lớn)
    ↓
child_chunks (Chunks nhỏ - indexed)
```

### Chi tiết từng tầng:

1. **SummaryDocument**: Tóm tắt ngắn gọn của document, dùng để xác định phạm vi tìm kiếm
2. **Document**: Tài liệu gốc, metadata và tracking
3. **ParentChunk**: Chunks lớn (context rộng) chứa nhiều child chunks
4. **ChildChunk**: Chunks nhỏ được indexed, dùng để tìm kiếm chi tiết

---

## 🔄 Workflow Hierarchical Retrieval (6 bước)

### **Bước 1: Tìm kiếm Summary Documents**

- Hybrid search (BM25 + Semantic) trên `summary_documents`
- Xác định phạm vi tài liệu liên quan
- **Fallback**: Nếu không tìm thấy hoặc score < 0.3 → tìm trực tiếp trên child chunks

### **Bước 2: Kiểm tra đủ thông tin**

- Format summary context và hỏi LLM: "Có đủ thông tin để trả lời không?"
- LLM trả về: `{sufficient: true/false, reason: "..."}`

### **Bước 3: Quyết định**

- **Nếu đủ**: Dùng summary để trả lời → Kết thúc
- **Nếu không đủ**: Tiến hành query expansion

### **Bước 4: Query Expansion**

- Trích xuất: entities, aliases, keywords
- Tạo query variants (biến thể câu hỏi)

### **Bước 5: Scoped Search trên Child Chunks**

- Tìm kiếm **chỉ trong phạm vi** child chunks thuộc summary docs đã tìm
- Dùng từng query variant
- Áp dụng RRF (Reciprocal Rank Fusion) để tổng hợp kết quả

### **Bước 6: Tạo câu trả lời**

- Gửi top chunks + câu hỏi cho LLM
- LLM tạo câu trả lời cuối cùng

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Copy environment template
copy .env.example .env

# Edit .env và thêm:
# GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Start Services

```bash
python -m src.main
```

Lưu ý set lại GEMINII_API_KEY trong file text_chunker.py

Bạn sẽ thấy menu với 5 tùy chọn:

```
================================================================================
CHỌN CHỨC NĂNG:
================================================================================
1. 📝 Chuẩn bị data
2. 🔪 Chunking và lưu vào Vector DB
3. 💬 RAG Chat (Interactive)
4. 🚀 Chạy cả hai (Chunking → Chat)
0. ❌ Thoát
================================================================================
```

## 📖 Hướng Dẫn Sử Dụng

### Option 1: 📝 Chuẩn Bị Data

**Chức năng:** Lấy dữ liệu từ Wikipedia và xử lý thành Markdown

**Các bước:**

1. Nhập từ khóa tìm kiếm (ví dụ: "Hồ Chí Minh", "Võ Nguyên Giáp")
2. Hệ thống sẽ:
   - Tải HTML từ Wikipedia
   - Làm sạch HTML
   - Chuyển đổi sang Markdown chuẩn hóa

**Kết quả:** File `.md` được lưu trong `data/processed_data/`

---

### Option 2: 🔪 Chunking và Lưu vào Vector DB

**Chức năng:** Tách văn bản thành chunks và lưu vào ChromaDB

**Cách hoạt động:**

- Đọc tất cả file `.md` trong `data/processed_data/`
- Tách theo Markdown headers (h1, h2)
- Lưu vào ChromaDB với embedding Google AI
- Tạo file pickle cho BM25 search

**Kết quả:**

- Vector DB: `data/chroma_db/`
- Pickle file: `data/chroma_db/knowledge_base_chunks.pkl`

**Collection name:** `knowledge_base` (mặc định)

---

## 📤 Upload Workflow

### **Bước 1: Upload Summary Document** (Bắt buộc trước)

```bash
curl -X POST "http://localhost:8000/api/v1/process-summary" \
  -F "files=@summary.html"
```

Response:

```json
{
  "total_files": 1,
  "results": [
    {
      "filename": "summary.html",
      "status": "processing",
      "job_id": "summary_abc123",
      "document_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  ]
}
```

**Lưu lại `document_id` (đây là `summary_id` cho bước sau)**

### **Bước 2: Upload Regular Documents**

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "summary_id=550e8400-e29b-41d4-a716-446655440000" \
  -F "files=@document1.html" \
  -F "files=@document2.html" \
  -F "chunk_size=800" \
  -F "chunk_overlap=150"
```

Response:

```json
{
  "total_files": 2,
  "results": [
    {
      "filename": "document1.html",
      "status": "processing",
      "job_id": "process_xyz789",
      "document_id": "660e8400-e29b-41d4-a716-446655440001"
    }
  ]
}
```

---

## 🔍 API Endpoints

### Upload Endpoints

| Endpoint                              | Method | Mô tả                    | Params                                                      |
| ------------------------------------- | ------ | ------------------------ | ----------------------------------------------------------- |
| `/api/v1/process-summary`             | POST   | Upload summary documents | files (multipart)                                           |
| `/api/v1/update-summary/{summary_id}` | POST   | Update summary document  | summary_id (path), file (multipart)                         |
| `/api/v1/process`                     | POST   | Upload regular documents | **summary_id** (required), files, chunk_size, chunk_overlap |

### Query Endpoints

| Endpoint       | Method | Mô tả                               |
| -------------- | ------ | ----------------------------------- |
| `/api/v1/chat` | POST   | RAG chat với hierarchical retrieval |

### Management Endpoints

| Endpoint                  | Method | Mô tả               |
| ------------------------- | ------ | ------------------- |
| `/api/v1/status/{job_id}` | GET    | Kiểm tra job status |
| `/api/v1/documents`       | GET    | List documents      |
| `/api/v1/documents/{id}`  | GET    | Get document detail |
| `/api/v1/documents/{id}`  | DELETE | Delete document     |

---

## 💬 RAG Chat Example

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Hồ Chí Minh sinh năm nào?",
    "document_ids": null,
    "verbose": true
  }'
```

Response:

```json
{
  "question": "Hồ Chí Minh sinh năm nào?",
  "answer": "Hồ Chí Minh sinh năm 1890 tại làng Kim Liên, Nghệ An.",
  "metadata": {
    "retrieval_method": "hierarchical",
    "summary_docs_found": 2,
    "sufficient_from_summaries": false,
    "query_variants": ["Năm sinh Hồ Chí Minh", "Bác Hồ sinh năm nào"],
    "child_chunks_retrieved": 5,
    "total_time_ms": 1234
  }
}
```

---

## 🗂️ Project Structure

```
RAG/
├── app/
│   ├── api/
│   │   ├── routes.py          # API endpoints
│   │   └── schemas.py         # Pydantic models
│   ├── database/
│   │   ├── models.py          # SQLAlchemy models (4-tier)
│   │   └── connection.py      # DB connection
│   ├── services/
│   │   ├── chunking_service.py    # Chunking logic
│   │   ├── embedding_service.py   # Gemini embeddings
│   │   ├── search_service.py      # Hybrid search (BM25 + Semantic)
│   │   └── rag_service.py         # Hierarchical RAG workflow
│   └── workers/
│       └── process_worker.py      # Background processing
├── migrations/
│   ├── create_bm25_index.sql              # ParadeDB BM25 index
│   └── update_embedding_dimension_768.sql # Update to 768-dim embeddings
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 📊 Database Schema

### summary_documents

```sql
- id (UUID, PK)
- file_path (TEXT)
- file_name (TEXT)
- content (TEXT)
- embedding (VECTOR(768))
- meta_data (JSONB)
- created_at (TIMESTAMP)
```

### documents

```sql
- id (UUID, PK)
- summary_id (UUID, FK → summary_documents)
- file_path (TEXT)
- file_name (TEXT)
- source_type (TEXT)
- status (TEXT)
- file_size (INTEGER)
- file_hash (TEXT, UNIQUE)
- meta_data (JSONB)
- created_at (TIMESTAMP)
```

### parent_chunks

```sql
- id (SERIAL, PK)
- document_id (UUID, FK → documents)
- summary_id (UUID, FK → summary_documents)
- content (TEXT)
- chunk_index (INTEGER)
- meta_data (JSONB)
- created_at (TIMESTAMP)
```

### child_chunks

```sql
- id (SERIAL, PK)
- document_id (UUID, FK → documents)
- parent_id (INTEGER, FK → parent_chunks)
- summary_id (UUID, FK → summary_documents)
- content (TEXT)
- embedding (VECTOR(768))
- chunk_index (INTEGER)
- section_id (INTEGER)
- h1, h2, h3 (TEXT)
- meta_data (JSONB)
- created_at (TIMESTAMP)
```

**Indexes:**

- ParadeDB BM25 index trên `child_chunks.content`
- pgvector HNSW index trên `child_chunks.embedding`
- pgvector HNSW index trên `summary_documents.embedding`

---

## ⚙️ Configuration

### Environment Variables (.env)

```env
# Gemini API
GEMINI_API_KEY=your_api_key_here

# Database
DATABASE_URL=postgresql://rag_user:rag_password@db:5432/rag_db

# Chunking
DEFAULT_CHUNK_SIZE=800
DEFAULT_CHUNK_OVERLAP=150

# Embedding
EMBEDDING_MODEL=models/text-embedding-004
EMBEDDING_DIMENSION=768
```

---

## 🎯 Key Features

### 1. **Hierarchical Retrieval**

- Tìm kiếm thông minh qua nhiều tầng
- Giảm nhiễu thông tin
- Tăng độ chính xác

### 2. **Scoped Search**

- Child chunks được giới hạn trong phạm vi summary documents
- Không tìm kiếm trên toàn bộ corpus → nhanh hơn

### 3. **Query Expansion**

- LLM tự động tạo query variants
- Trích xuất entities, aliases
- Tăng recall rate

### 4. **Hybrid Search**

- **BM25** (ParadeDB): Keyword matching
- **Semantic** (pgvector): Cosine similarity
- **RRF**: Fusion algorithm tổng hợp kết quả

### 5. **Background Processing**

- Upload files → trả về job_id ngay lập tức
- Worker xử lý background (ingest → chunk → embed → save)

---

## 📝 Notes

### Upload Order

⚠️ **Bắt buộc**: Upload summary documents trước, sau đó mới upload regular documents với `summary_id`

### Query Variants

LLM tự động tạo biến thể câu hỏi để tăng khả năng tìm kiếm

### Fallback Mechanism

Nếu không tìm thấy summary docs hoặc score thấp → tìm kiếm trực tiếp trên toàn bộ child chunks

---

## 🚀 Production Tips

1. **Monitoring**: Thêm Prometheus + Grafana để monitor performance
2. **Load Balancing**: Scale database replicas cho heavy loads
3. **Backup**: Định kỳ backup PostgreSQL database
4. **Rate Limiting**: Thêm rate limiter cho API endpoints

---

## 📄 License

MIT License
