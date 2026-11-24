FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf DATA_DIR=/data PORT=8080
WORKDIR /app

# CPU-only torch (the detector runs on CPU; the CUDA wheels would add ~4 GB)
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml /app/
RUN pip install --no-cache-dir "transformers==4.55.2" "huggingface_hub>=0.34" numpy "httpx>=0.27" "anthropic>=0.70" "google-genai>=1.30" \
    "fastapi>=0.115" "uvicorn[standard]>=0.30" python-multipart "authlib>=1.3" itsdangerous python-docx pypdf

# The error detector is public on Hugging Face; bake it into the image so startup needs no network or token.
ARG DETECTOR_MODEL=Sefaria/BEREL_3.0-hebrew-error-detector
ARG DETECTOR_REVISION=main
RUN python -c "from huggingface_hub import snapshot_download; print(snapshot_download('${DETECTOR_MODEL}', revision='${DETECTOR_REVISION}'))" \
    && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1

COPY pipeline /app/pipeline
COPY app /app/app
COPY web /app/web
RUN useradd --uid 10001 --create-home corrector && mkdir -p /data && chown -R corrector:corrector /data
USER corrector
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"
# One process: the background worker threads live inside the web process, so run exactly one replica and one worker.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
