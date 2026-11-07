FROM python:3.13-slim AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY backend/requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.13-slim
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN useradd -m -u 10001 app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY backend ./backend
COPY run_server.py .
RUN mkdir -p /data && chown -R app:app /app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["python", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]
