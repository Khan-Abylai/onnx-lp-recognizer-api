FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
  && apt-get install -y ssh \
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
    python-is-python3 2to3 \
    python3-dev \
    python3-pip \
    ffmpeg libsm6 libxext6 libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgtk-3-dev\
    libpng-dev libopenexr-dev libtiff-dev libwebp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    RUN pip install --upgrade pip
    RUN pip install tqdm pydantic_settings llvmlite scipy opencv-python uvicorn fastapi python-multipart scikit-image pycocotools shortuuid onnxruntime onnx torch torchvision pydantic 
# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]