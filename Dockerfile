FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies for PDF processing, SQLite, and network tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Pre-create data directories
RUN mkdir -p data/resumes data/reports data/vectordb

# Expose public and API ports
EXPOSE 8501 8000

# Default environment variables
ENV API_BASE_URL="http://127.0.0.1:8000" \
    PORT=8501

# Run the unified cloud deployment runner
CMD ["python", "deploy_runner.py"]
