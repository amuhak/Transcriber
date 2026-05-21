# Use the official, highly-optimized vLLM image
FROM vllm/vllm-openai:latest

# Install system dependencies (ffmpeg for audio extraction)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install additional Python speech processing dependencies if not present
RUN pip install --no-cache-dir librosa soundfile

# Pre-download the model weights so the image is 100% portable and offline-ready
RUN hf download ibm-granite/granite-speech-4.1-2b

# Copy our vLLM transcription script into the container
COPY transcribe.py .

# Clear the base image's entrypoint so we can run custom python commands directly
ENTRYPOINT []
