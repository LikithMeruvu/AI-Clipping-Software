FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY clipper ./clipper
COPY main.py pyproject.toml ./
ENV DATA_DIR=/data OUTPUT_DIR=/data/output WHISPER_DEVICE=auto HOST=0.0.0.0
VOLUME /data
EXPOSE 8000
# API_TOKEN must be set when binding to 0.0.0.0
CMD ["python", "-m", "clipper", "serve"]
