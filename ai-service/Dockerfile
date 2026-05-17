FROM python:3.10-slim

# System deps: ffmpeg for video encode/decode, libgl/libglib for opencv runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Hugging Face Spaces runs containers as a non-root user with UID 1000 and
# expects a writable home, so set it up explicitly.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user app/ app/

# HF Spaces routes incoming traffic to port 7860 by default.
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
