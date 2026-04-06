FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including Java for VnCoreNLP)
# Use --no-install-recommends to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    postgresql-client \
    default-jre-headless \
    wget \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Pre-download VnCoreNLP JAR and word-segmenter models
# Cache this layer separately for faster rebuilds
RUN mkdir -p /app/vncorenlp/models/wordsegmenter \
    && wget -q -O /app/vncorenlp/VnCoreNLP-1.2.jar \
    https://github.com/vncorenlp/VnCoreNLP/raw/master/VnCoreNLP-1.2.jar \
    && wget -q -O /app/vncorenlp/models/wordsegmenter/vi-vocab \
    https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab \
    && wget -q -O /app/vncorenlp/models/wordsegmenter/wordsegmenter.rdr \
    https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr

# Copy only requirements first (for better cache layering)
COPY requirements.txt .

# Install Python dependencies with PIP optimization
# Use --no-cache-dir to reduce final image size
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# Copy application code
COPY . .

# Create data directory (only uploads needed)
RUN mkdir -p /app/data/uploads && rm -rf /app/data/temp_uploads

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Set VnCoreNLP socket timeout (60 seconds)
ENV VNCORENLP_SOCKET_TIMEOUT=60000

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
