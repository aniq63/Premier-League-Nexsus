# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables required for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (useful for building packages like lxml, psycopg2, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project first!
# This is CRITICAL because your requirements.txt contains `-e .`
# Pip needs `setup.py`, `pyproject.toml`, and the `src` directory present to successfully install it as an SDK
COPY . /app/

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Run the FastAPI application using uvicorn
# Railway uses the PORT environment variable dynamically
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
