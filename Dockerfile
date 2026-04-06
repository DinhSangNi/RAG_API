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

# Create data directory (only uploads needed)
RUN mkdir -p /app/data/uploads && rm -rf /app/data/temp_uploads

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
