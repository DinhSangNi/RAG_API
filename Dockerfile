FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# Use --no-install-recommends to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only requirements first (for better cache layering)
COPY requirements.txt .

# Install Python dependencies with PIP optimization
# Use --no-cache-dir to reduce final image size
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# Copy application code
COPY . .

# Ensure shared storage directory exists with proper permissions
# This may be overridden by Volume mounts, so use -p to not fail if it exists
# RUN mkdir -p /app/data && chmod -R 777 /app/data || true
RUN mkdir -p /home/site/temp_data && ln -s /home/site/temp_data /app/data

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
