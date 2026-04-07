FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Redis
# Use --no-install-recommends to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    postgresql-client \
    redis-server \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only requirements first (for better cache layering)
COPY requirements.txt .

# Install Python dependencies with PIP optimization
# Use --no-cache-dir to reduce final image size
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# Copy application code
COPY . .

# Copy and make entrypoint script executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create data directory
RUN mkdir -p /app/data/temp_uploads && chmod -R 777 /app/data

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Expose ports
# 8000: FastAPI
# 6379: Redis
EXPOSE 8000 6379

# Use shell script to start all services
ENTRYPOINT ["/app/entrypoint.sh"]
