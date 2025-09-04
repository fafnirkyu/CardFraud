# Use Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI and MLflow ports
EXPOSE 8000
EXPOSE 5000

# Start both MLflow and FastAPI using supervisord (or just one for now)
CMD uvicorn app:app --host 0.0.0.0 --port 8000
