FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including Java for VnCoreNLP)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    default-jre-headless \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Pre-download VnCoreNLP JAR and word-segmenter models
RUN mkdir -p /app/vncorenlp/models/wordsegmenter \
    && wget -q -O /app/vncorenlp/VnCoreNLP-1.2.jar \
        https://github.com/vncorenlp/VnCoreNLP/raw/master/VnCoreNLP-1.2.jar \
    && wget -q -O /app/vncorenlp/models/wordsegmenter/vi-vocab \
        https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab \
    && wget -q -O /app/vncorenlp/models/wordsegmenter/wordsegmenter.rdr \
        https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/processed_data /app/data/raw_data

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
