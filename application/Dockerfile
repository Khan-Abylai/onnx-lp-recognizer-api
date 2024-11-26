FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update -o Acquire::http::Pipeline-Depth=200 && \
    apt-get install -y --no-install-recommends \
    ssh \
    build-essential \
    gcc \
    g++ \
    gdb \
    clang \
    make \
    ninja-build \
    cmake \
    autoconf \
    automake \
    locales-all \
    dos2unix \
    rsync \
    tar \
    python-is-python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libpng-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Expose port for the application
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "debug"]
